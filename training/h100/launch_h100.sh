#!/usr/bin/env bash
# One-command launcher for the LexBrowser WebVoyager GRPO recipe on H100.
#
# Launcher chain of the validated recipe with the
# validated 2026-07-21 hyperparameters baked in as defaults. This subtree is
# self-contained: every referenced script and runtime file lives under
# training/h100/.
#
# Single node (8x H100):
#   NODES_CSV=<this-host-ip> bash training/h100/launch_h100.sh
# Two nodes (16x H100, same world size as the validated run):
#   NODES_CSV=<head-ip>,<worker-ip> bash training/h100/launch_h100.sh
#
# Requirements before launching: see training/h100/README.md.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$SCRIPT_DIR}
WORK_ROOT=${WORK_ROOT:-/data/lexbrowser-rl}
RUNS_ROOT=${RUNS_ROOT:-$WORK_ROOT/runs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$WORK_ROOT/checkpoints}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}
IMAGE=${IMAGE:-lexbrowser-verl-h100:local}
RAY_CONTAINER=${RAY_CONTAINER:-lexbrowser-h100-ray}
# Ray control-plane ports. Default is ray's own 6379/8265; override when a
# foreign ray cluster (e.g. a colleague's webagent-ray) already holds them —
# joining someone else's GCS fails with a session-name assertion.
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

# Node topology. NODES_CSV lists node IPs, head first.
NODES_CSV=${NODES_CSV:-$(hostname -I 2>/dev/null | awk '{print $1}')}
if [[ -z "$NODES_CSV" ]]; then
  echo "Could not auto-detect this host's IP; set NODES_CSV explicitly." >&2
  exit 2
fi
IFS=',' read -r -a NODES <<<"$NODES_CSV"
HEAD_IP=${HEAD_IP:-${NODES[0]}}
NNODES=${#NODES[@]}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_USER=${SSH_USER:-root}
SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

# Browser environment sidecar.
NEMO_GYM_PORT=${NEMO_GYM_PORT:-18180}
MAX_CONCURRENT_SESSIONS=${MAX_CONCURRENT_SESSIONS:-64}
MAX_CONCURRENT_CREATES=${MAX_CONCURRENT_CREATES:-16}
BROWSER_BACKEND=${BROWSER_BACKEND:-lexmount}
# "normal" = Chrome-based cloud browser, "light" = lightmount engine.
LEXMOUNT_BROWSER_MODE=${LEXMOUNT_BROWSER_MODE:-normal}
LEXBROWSER_JUDGE_FALLBACK_MODELS=${LEXBROWSER_JUDGE_FALLBACK_MODELS:-}
LEXMOUNT_SESSION_CREATE_TIMEOUT_S=${LEXMOUNT_SESSION_CREATE_TIMEOUT_S:-60}
LEXMOUNT_RESET_TOTAL_BUDGET_S=${LEXMOUNT_RESET_TOTAL_BUDGET_S:-110}
LOCAL_CDP_HTTP_URL=${LOCAL_CDP_HTTP_URL:-http://127.0.0.1:9222}

# --- Validated hyperparameters (identical to the 2026-07-21 60-step run). ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
TOTAL_STEPS=${TOTAL_STEPS:-60}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-40960}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-36864}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-10}
MAX_USER_TURNS=${MAX_USER_TURNS:-10}
ACTION_MAX_TOKENS=${ACTION_MAX_TOKENS:-1024}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-16384}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}
# 12288/GPU was the validated packing budget on 64 GB devices; 15360 = 12288 * 80/64.
# Packing-only knob; see training/h100/README.md before changing.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-15360}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PPO_MAX_TOKEN_LEN_PER_GPU}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PPO_MAX_TOKEN_LEN_PER_GPU}
ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true}
ENTROPY_FROM_LOGITS_CHUNK_SIZE=${ENTROPY_FROM_LOGITS_CHUNK_SIZE:-256}
VERL_PROCESS_GROUP_TIMEOUT_SECONDS=${VERL_PROCESS_GROUP_TIMEOUT_SECONDS:-7200}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)-h100-40k10t}
RUN_DIR=$RUNS_ROOT/$STAMP
# In-container paths (repo is mounted at /workspace/lexbrowser-h100).
DATA=${DATA:-/workspace/lexbrowser-h100/data/webvoyager-clean/train.lexbrowser.parquet}
HOST_DATA=${HOST_DATA:-$ROOT/data/webvoyager-clean/train.lexbrowser.parquet}
AUDIT_DIR=${AUDIT_DIR:-$RUN_DIR/audit}
LEXBROWSER_METRICS_DIR=/workspace/runs/$STAMP/observability/raw
# Container paths for the Ray-node env: Ray actors inherit the ray-node
# container's environment, not the trainer driver's exports, so these must be
# injected at `docker run` time or TensorBoard/metrics land on an unmounted
# relative path inside the container.
TENSORBOARD_DIR=/workspace/runs/$STAMP/tensorboard
VERL_FILE_LOGGER_ROOT=/workspace/runs/$STAMP/metrics

export ROOT RUNS_ROOT CHECKPOINT_ROOT NODES_CSV HEAD_IP STAMP IMAGE
export TENSORBOARD_DIR VERL_FILE_LOGGER_ROOT
export WORK_ROOT MODEL_PATH GPUS_PER_NODE
export BROWSER_BACKEND LOCAL_CDP_HTTP_URL

if [[ "$TRAIN_BATCH_SIZE" -ne 8 || "$ROLLOUT_N" -ne 8 || "$PPO_MINI_BATCH_SIZE" -ne 8 ]]; then
  # The 2026-07-21 60-step run validated 8/8/8, and a silent change of geometry
  # invalidates any comparison against it, so deviating has to be deliberate.
  # Diagnostic runs legitimately need a smaller batch: two engines can then be
  # compared side by side on separate nodes without exceeding the browser
  # account's 80-session ceiling.
  if [[ "${ALLOW_UNVALIDATED_GEOMETRY:-0}" != "1" ]]; then
    echo "The validated TrainerV1 geometry is train_batch_size=8, rollout.n=8, ppo_mini_batch_size=8." >&2
    echo "Set ALLOW_UNVALIDATED_GEOMETRY=1 to run a different one; results are not comparable with the validated run." >&2
    exit 2
  fi
  echo "GEOMETRY_OVERRIDE train_batch_size=$TRAIN_BATCH_SIZE rollout_n=$ROLLOUT_N ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
fi
# run_lexbrowser_grpo_h100.sh carries its own geometry guard, comparing against
# EXPECTED_TASKS_PER_STEP / EXPECTED_ROLLOUTS_PER_TASK. Both are env-overridable
# there by design, but nothing forwarded them, so an override accepted here was
# rejected inside the container one line into the trainer. Carry the intent
# through instead of leaving two guards that disagree.
EXPECTED_TASKS_PER_STEP=${EXPECTED_TASKS_PER_STEP:-$TRAIN_BATCH_SIZE}
EXPECTED_ROLLOUTS_PER_TASK=${EXPECTED_ROLLOUTS_PER_TASK:-$ROLLOUT_N}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-True}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
DYNAMIC_SAMPLING=${DYNAMIC_SAMPLING:-1}
GROUP_RESAMPLES=${GROUP_RESAMPLES:-2}
if (( 4096 + MAX_RESPONSE_LENGTH > MAX_MODEL_LENGTH )); then
  echo "initial prompt (4096) + rollout ($MAX_RESPONSE_LENGTH) exceeds model length ($MAX_MODEL_LENGTH)." >&2
  exit 2
fi

# Build the 168-task parquet on first use (deterministic; see MANIFEST.json).
if [[ ! -s "$HOST_DATA" ]]; then
  echo "Building webvoyager-clean training parquet..."
  docker run --rm --network none \
    -v "$ROOT:/workspace/lexbrowser-h100" \
    -w /workspace/lexbrowser-h100 --entrypoint python3 "$IMAGE" \
    build_webvoyager_clean_data.py
fi
test -s "$HOST_DATA"
echo "TASK_DATA_OK sha256=$(sha256sum "$HOST_DATA" | awk '{print $1}')"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  NODES_CSV="$NODES_CSV" HEAD_IP="$HEAD_IP" bash "$ROOT/preflight_h100.sh"
fi

PORT="$NEMO_GYM_PORT" MAX_CONCURRENT_SESSIONS="$MAX_CONCURRENT_SESSIONS" \
  MAX_CONCURRENT_CREATES="$MAX_CONCURRENT_CREATES" \
  BROWSER_BACKEND="$BROWSER_BACKEND" LOCAL_CDP_HTTP_URL="$LOCAL_CDP_HTTP_URL" \
  LEXMOUNT_BROWSER_MODE="$LEXMOUNT_BROWSER_MODE" \
  LEXBROWSER_JUDGE_FALLBACK_MODELS="$LEXBROWSER_JUDGE_FALLBACK_MODELS" \
  LEXMOUNT_SESSION_CREATE_TIMEOUT_S="$LEXMOUNT_SESSION_CREATE_TIMEOUT_S" \
  LEXMOUNT_RESET_TOTAL_BUDGET_S="$LEXMOUNT_RESET_TOTAL_BUDGET_S" \
  AUDIT_DIR="$AUDIT_DIR" \
  bash "$ROOT/start_nemo_gym_webvoyager_server_h100.sh"
export NEMO_GYM_BROWSER_URL="http://$HEAD_IP:$NEMO_GYM_PORT"

mkdir -p "$RUN_DIR/logs" "$CHECKPOINT_ROOT/$STAMP"

ROLE=head NODE_IP="$HEAD_IP" HEAD_IP="$HEAD_IP" ROOT="$ROOT" MODEL_PATH="$MODEL_PATH" \
  RAY_PORT="$RAY_PORT" RAY_DASHBOARD_PORT="$RAY_DASHBOARD_PORT" \
  RUNS_ROOT="$RUNS_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" IMAGE="$IMAGE" NAME="$RAY_CONTAINER" \
  VERL_PROCESS_GROUP_TIMEOUT_SECONDS="$VERL_PROCESS_GROUP_TIMEOUT_SECONDS" \
  LEXBROWSER_ACTION_MAX_TOKENS="$ACTION_MAX_TOKENS" \
  LEXBROWSER_DYNAMIC_SAMPLING="$DYNAMIC_SAMPLING" \
  LEXBROWSER_GROUP_RESAMPLES="$GROUP_RESAMPLES" \
  LEXBROWSER_METRICS_DIR="$LEXBROWSER_METRICS_DIR" \
  NEMO_GYM_BROWSER_URL="$NEMO_GYM_BROWSER_URL" \
  bash "$ROOT/start_ray_node_h100.sh"

head_ready=0
for _ in $(seq 1 120); do
  if timeout 2 bash -c "</dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null; then
    head_ready=1
    break
  fi
  sleep 1
done
if [[ "$head_ready" != 1 ]]; then
  echo "Ray head GCS did not listen on $HEAD_IP:$RAY_PORT" >&2
  exit 1
fi

for node in "${NODES[@]}"; do
  [[ "$node" == "$HEAD_IP" ]] && continue
  "${SSH[@]}" "$SSH_USER@$node" "ROLE=worker NODE_IP=$node HEAD_IP=$HEAD_IP ROOT=$ROOT RAY_PORT=$RAY_PORT RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT RUNS_ROOT=$RUNS_ROOT CHECKPOINT_ROOT=$CHECKPOINT_ROOT MODEL_PATH=$MODEL_PATH IMAGE=$IMAGE NAME=$RAY_CONTAINER NEMO_GYM_BROWSER_URL=$NEMO_GYM_BROWSER_URL LEXBROWSER_ACTION_MAX_TOKENS=$ACTION_MAX_TOKENS LEXBROWSER_DYNAMIC_SAMPLING=$DYNAMIC_SAMPLING LEXBROWSER_GROUP_RESAMPLES=$GROUP_RESAMPLES LEXBROWSER_METRICS_DIR=$LEXBROWSER_METRICS_DIR TENSORBOARD_DIR=$TENSORBOARD_DIR VERL_FILE_LOGGER_ROOT=$VERL_FILE_LOGGER_ROOT VERL_PROCESS_GROUP_TIMEOUT_SECONDS=$VERL_PROCESS_GROUP_TIMEOUT_SECONDS bash $ROOT/start_ray_node_h100.sh"
done

expected_gpus=$((NNODES * GPUS_PER_NODE))
cluster_ready=0
for _ in $(seq 1 60); do
  if timeout 15 docker exec "$RAY_CONTAINER" python3 -c \
    "import ray; ray.init(address='$HEAD_IP:$RAY_PORT'); resources=ray.cluster_resources(); actual=int(resources.get('GPU', 0)); print(f'RAY_CLUSTER_GPU actual={actual} expected=$expected_gpus'); assert actual == $expected_gpus"; then
    cluster_ready=1
    break
  fi
  sleep 3
done
if [[ "$cluster_ready" != 1 ]]; then
  echo "Ray cluster did not reach $expected_gpus GPU resources" >&2
  docker exec "$RAY_CONTAINER" ray status --address="$HEAD_IP:$RAY_PORT" || true
  exit 1
fi

docker exec -d "$RAY_CONTAINER" bash -lc \
  "cd /workspace/lexbrowser-h100 && mkdir -p /workspace/runs/$STAMP/logs /workspace/checkpoints/$STAMP && RUNS_ROOT=/workspace/runs CHECKPOINT_ROOT=/workspace/checkpoints RESUME_FROM_PATH=$RESUME_FROM_PATH VERL_PROCESS_GROUP_TIMEOUT_SECONDS=$VERL_PROCESS_GROUP_TIMEOUT_SECONDS STAMP=$STAMP MODEL_PATH=$MODEL_PATH DATA=$DATA NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE ROLLOUT_N=$ROLLOUT_N PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE TOTAL_STEPS=$TOTAL_STEPS TOTAL_EPOCHS=$TOTAL_EPOCHS SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH MAX_MODEL_LENGTH=$MAX_MODEL_LENGTH ULYSSES_SEQUENCE_PARALLEL_SIZE=$ULYSSES_SEQUENCE_PARALLEL_SIZE PPO_MAX_TOKEN_LEN_PER_GPU=$PPO_MAX_TOKEN_LEN_PER_GPU REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU ENTROPY_FROM_LOGITS_WITH_CHUNKING=$ENTROPY_FROM_LOGITS_WITH_CHUNKING ENTROPY_FROM_LOGITS_CHUNK_SIZE=$ENTROPY_FROM_LOGITS_CHUNK_SIZE MAX_ASSISTANT_TURNS=$MAX_ASSISTANT_TURNS MAX_USER_TURNS=$MAX_USER_TURNS MAX_TOOL_RESPONSE_LENGTH=$MAX_TOOL_RESPONSE_LENGTH REASONING_PARSER=$REASONING_PARSER EXPECTED_TASKS_PER_STEP=$EXPECTED_TASKS_PER_STEP EXPECTED_ROLLOUTS_PER_TASK=$EXPECTED_ROLLOUTS_PER_TASK CLIP_RATIO_LOW=$CLIP_RATIO_LOW CLIP_RATIO_HIGH=$CLIP_RATIO_HIGH CLIP_RATIO_C=$CLIP_RATIO_C USE_KL_LOSS=$USE_KL_LOSS KL_LOSS_COEF=$KL_LOSS_COEF NORM_ADV_BY_STD=$NORM_ADV_BY_STD LOSS_AGG_MODE=$LOSS_AGG_MODE DYNAMIC_SAMPLING=$DYNAMIC_SAMPLING GROUP_RESAMPLES=$GROUP_RESAMPLES bash run_lexbrowser_grpo_h100.sh > /workspace/runs/$STAMP/logs/train.log 2>&1"

cat <<EOF
LAUNCH_OK run_id=$STAMP nodes=$NNODES gpus=$expected_gpus backend=$BROWSER_BACKEND browser_mode=$LEXMOUNT_BROWSER_MODE
Training log:   $RUN_DIR/logs/train.log
TensorBoard:    $RUN_DIR/tensorboard  (tensorboard --logdir $RUN_DIR/tensorboard)
Rollout audit:  $RUN_DIR/rollouts
Checkpoints:    $CHECKPOINT_ROOT/$STAMP (every $SAVE_FREQ steps)
Sidecar logs:   docker logs -f lexbrowser-nemo-gym-webvoyager
Ray status:     docker exec $RAY_CONTAINER ray status --address=$HEAD_IP:$RAY_PORT
EOF
