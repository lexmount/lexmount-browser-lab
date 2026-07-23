#!/usr/bin/env bash
# One-command launcher for the LexBrowser WebVoyager GRPO recipe on H100.
#
# Mirrors the internal Ascend launcher chain of the validated recipe with the
# validated 2026-07-21 hyperparameters baked in as defaults. This subtree is
# self-contained: every referenced script and runtime file lives under
# training/h100/.
#
# Single node (8x H100):
#   NODES_CSV=<this-host-ip> bash training/h100/launch_h100.sh
# Two nodes (16x H100, same world size as the validated 2x8 910B run):
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
# 12288/GPU was validated on 64 GB 910B; 15360 = 12288 * 80/64 for H100-80GB.
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

export ROOT RUNS_ROOT CHECKPOINT_ROOT NODES_CSV HEAD_IP STAMP IMAGE
export WORK_ROOT MODEL_PATH GPUS_PER_NODE
export BROWSER_BACKEND LOCAL_CDP_HTTP_URL

if [[ "$TRAIN_BATCH_SIZE" -ne 8 || "$ROLLOUT_N" -ne 8 || "$PPO_MINI_BATCH_SIZE" -ne 8 ]]; then
  echo "The validated TrainerV1 geometry is train_batch_size=8, rollout.n=8, ppo_mini_batch_size=8." >&2
  exit 2
fi
if (( 4096 + MAX_RESPONSE_LENGTH > MAX_MODEL_LENGTH )); then
  echo "initial prompt (4096) + rollout ($MAX_RESPONSE_LENGTH) exceeds model length ($MAX_MODEL_LENGTH)." >&2
  exit 2
fi

# Build the 168-task parquet on first use (deterministic; see MANIFEST.json).
if [[ ! -s "$HOST_DATA" ]]; then
  echo "Building webvoyager-clean training parquet..."
  docker run --rm \
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
  AUDIT_DIR="$AUDIT_DIR" \
  bash "$ROOT/start_nemo_gym_webvoyager_server_h100.sh"
export NEMO_GYM_BROWSER_URL="http://$HEAD_IP:$NEMO_GYM_PORT"

mkdir -p "$RUN_DIR/logs" "$CHECKPOINT_ROOT/$STAMP"

ROLE=head NODE_IP="$HEAD_IP" HEAD_IP="$HEAD_IP" ROOT="$ROOT" MODEL_PATH="$MODEL_PATH" \
  RUNS_ROOT="$RUNS_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" IMAGE="$IMAGE" NAME="$RAY_CONTAINER" \
  VERL_PROCESS_GROUP_TIMEOUT_SECONDS="$VERL_PROCESS_GROUP_TIMEOUT_SECONDS" \
  LEXBROWSER_ACTION_MAX_TOKENS="$ACTION_MAX_TOKENS" \
  LEXBROWSER_METRICS_DIR="$LEXBROWSER_METRICS_DIR" \
  NEMO_GYM_BROWSER_URL="$NEMO_GYM_BROWSER_URL" \
  bash "$ROOT/start_ray_node_h100.sh"

head_ready=0
for _ in $(seq 1 120); do
  if timeout 2 bash -c "</dev/tcp/$HEAD_IP/6379" 2>/dev/null; then
    head_ready=1
    break
  fi
  sleep 1
done
if [[ "$head_ready" != 1 ]]; then
  echo "Ray head GCS did not listen on $HEAD_IP:6379" >&2
  exit 1
fi

for node in "${NODES[@]}"; do
  [[ "$node" == "$HEAD_IP" ]] && continue
  "${SSH[@]}" "$SSH_USER@$node" "ROLE=worker NODE_IP=$node HEAD_IP=$HEAD_IP ROOT=$ROOT RUNS_ROOT=$RUNS_ROOT CHECKPOINT_ROOT=$CHECKPOINT_ROOT MODEL_PATH=$MODEL_PATH IMAGE=$IMAGE NAME=$RAY_CONTAINER NEMO_GYM_BROWSER_URL=$NEMO_GYM_BROWSER_URL LEXBROWSER_ACTION_MAX_TOKENS=$ACTION_MAX_TOKENS LEXBROWSER_METRICS_DIR=$LEXBROWSER_METRICS_DIR VERL_PROCESS_GROUP_TIMEOUT_SECONDS=$VERL_PROCESS_GROUP_TIMEOUT_SECONDS bash $ROOT/start_ray_node_h100.sh"
done

expected_gpus=$((NNODES * GPUS_PER_NODE))
cluster_ready=0
for _ in $(seq 1 60); do
  if timeout 15 docker exec "$RAY_CONTAINER" python3 -c \
    "import ray; ray.init(address='$HEAD_IP:6379'); resources=ray.cluster_resources(); actual=int(resources.get('GPU', 0)); print(f'RAY_CLUSTER_GPU actual={actual} expected=$expected_gpus'); assert actual == $expected_gpus"; then
    cluster_ready=1
    break
  fi
  sleep 3
done
if [[ "$cluster_ready" != 1 ]]; then
  echo "Ray cluster did not reach $expected_gpus GPU resources" >&2
  docker exec "$RAY_CONTAINER" ray status --address="$HEAD_IP:6379" || true
  exit 1
fi

docker exec -d "$RAY_CONTAINER" bash -lc \
  "cd /workspace/lexbrowser-h100 && mkdir -p /workspace/runs/$STAMP/logs /workspace/checkpoints/$STAMP && RUNS_ROOT=/workspace/runs CHECKPOINT_ROOT=/workspace/checkpoints RESUME_FROM_PATH=$RESUME_FROM_PATH VERL_PROCESS_GROUP_TIMEOUT_SECONDS=$VERL_PROCESS_GROUP_TIMEOUT_SECONDS STAMP=$STAMP MODEL_PATH=$MODEL_PATH DATA=$DATA NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE ROLLOUT_N=$ROLLOUT_N PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE TOTAL_STEPS=$TOTAL_STEPS TOTAL_EPOCHS=$TOTAL_EPOCHS SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH MAX_MODEL_LENGTH=$MAX_MODEL_LENGTH ULYSSES_SEQUENCE_PARALLEL_SIZE=$ULYSSES_SEQUENCE_PARALLEL_SIZE PPO_MAX_TOKEN_LEN_PER_GPU=$PPO_MAX_TOKEN_LEN_PER_GPU REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU ENTROPY_FROM_LOGITS_WITH_CHUNKING=$ENTROPY_FROM_LOGITS_WITH_CHUNKING ENTROPY_FROM_LOGITS_CHUNK_SIZE=$ENTROPY_FROM_LOGITS_CHUNK_SIZE MAX_ASSISTANT_TURNS=$MAX_ASSISTANT_TURNS MAX_USER_TURNS=$MAX_USER_TURNS MAX_TOOL_RESPONSE_LENGTH=$MAX_TOOL_RESPONSE_LENGTH REASONING_PARSER=$REASONING_PARSER bash run_lexbrowser_grpo_h100.sh > /workspace/runs/$STAMP/logs/train.log 2>&1"

cat <<EOF
LAUNCH_OK run_id=$STAMP nodes=$NNODES gpus=$expected_gpus backend=$BROWSER_BACKEND
Training log:   $RUN_DIR/logs/train.log
TensorBoard:    $RUN_DIR/tensorboard  (tensorboard --logdir $RUN_DIR/tensorboard)
Rollout audit:  $RUN_DIR/rollouts
Checkpoints:    $CHECKPOINT_ROOT/$STAMP (every $SAVE_FREQ steps)
Sidecar logs:   docker logs -f lexbrowser-nemo-gym-webvoyager
Ray status:     docker exec $RAY_CONTAINER ray status --address=$HEAD_IP:6379
EOF
