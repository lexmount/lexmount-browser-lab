#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(dirname "$ROOT")"
IMAGE="${NEMO_RL_IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"
MODEL_DIR="${MODEL_DIR:-/home/wf/models/Qwen3-1.7B}"
MODE="${1:-train}"
SMOKE_TASK_OFFSET="${WEBVOYAGER_SMOKE_TASK_OFFSET:-0}"
DATA_SOURCE="$ROOT/training/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/datasets/WebVoyager_data_clean.jsonl"
DATA_DIR="$ROOT/training/data/webvoyager"
GYM_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/verifiers_agent"
GYM_MODEL_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model"
GYM_CACHE_DIR="$WORKSPACE/.cache/nemo-rl/gym-venvs"

if [[ "$MODE" != "train" && "$MODE" != "smoke" && "$MODE" != "stage1" && "$MODE" != "stage2" ]]; then
  echo "usage: $0 [train|smoke|stage1|stage2]" >&2
  exit 2
fi

for path in "$ROOT/secrets.env" "$MODEL_DIR" "$DATA_SOURCE"; do
  if [[ ! -e "$path" ]]; then
    echo "required path is missing: $path" >&2
    exit 1
  fi
done

if [[ "$(stat -c '%a' "$ROOT/secrets.env")" != "600" ]]; then
  echo "secrets.env must have mode 600" >&2
  exit 1
fi

for name in LEXMOUNT_API_KEY LEXMOUNT_PROJECT_ID OPENAI_API_KEY OPENAI_BASE_URL; do
  if ! grep -q "^${name}=" "$ROOT/secrets.env"; then
    echo "secrets.env is missing $name" >&2
    exit 1
  fi
done

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -lt 2 ]]; then
  echo "two visible NVIDIA GPUs are required" >&2
  exit 1
fi

mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
for index in 0 1; do
  if (( ${free_mib[$index]} < 22000 )); then
    echo "GPU $index has only ${free_mib[$index]} MiB free; at least 22000 MiB is required" >&2
    exit 1
  fi
done

mkdir -p "$DATA_DIR" "$ROOT/logs/lexbrowser-grpo" "$ROOT/results/lexbrowser-grpo" \
  "$WORKSPACE/.cache/nemo-rl" "$GYM_CACHE_DIR/verifiers-agent" "$GYM_CACHE_DIR/vllm-model" || true
# Docker can leave TensorBoard/log directories root-owned after an interrupted
# run.  The preflight must be able to write its own evidence before GPU work
# starts, so repair ownership only for this experiment's generated directories.
if ! touch "$ROOT/logs/lexbrowser-grpo/.lexbrowser_write_probe" 2>/dev/null; then
  sudo -n chown -R "$(id -u):$(id -g)" \
    "$ROOT/logs/lexbrowser-grpo" "$ROOT/results/lexbrowser-grpo" "$ROOT/training/data"
  touch "$ROOT/logs/lexbrowser-grpo/.lexbrowser_write_probe"
fi
rm -f "$ROOT/logs/lexbrowser-grpo/.lexbrowser_write_probe"
python3 "$ROOT/training/scripts/prepare_webvoyager_data.py" \
  --source "$DATA_SOURCE" \
  --output "$DATA_DIR/train.jsonl" \
  --manifest "$DATA_DIR/manifest.json"
python3 "$ROOT/training/scripts/prepare_webvoyager_data.py" \
  --source "$DATA_SOURCE" \
  --output "$DATA_DIR/smoke.jsonl" \
  --manifest "$DATA_DIR/smoke-manifest.json" \
  --limit 1 \
  --offset "$SMOKE_TASK_OFFSET"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  sudo -v
  DOCKER=(sudo docker)
fi

if ! "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
  "${DOCKER[@]}" pull "$IMAGE"
fi

uid="$(id -u)"
gid="$(id -g)"
container_name="lexbrowser-grpo-${MODE}-$$"
restore_ownership() {
  "${DOCKER[@]}" rm -f "$container_name" >/dev/null 2>&1 || true
  sudo -n chown -R "$uid:$gid" \
    "$ROOT/logs/lexbrowser-grpo" \
    "$ROOT/results/lexbrowser-grpo" \
    "$ROOT/training/data" >/dev/null 2>&1 || true
}
trap restore_ownership EXIT

overrides=()
if [[ "$MODE" == "smoke" ]]; then
  overrides=(
    grpo.num_prompts_per_step=1
    grpo.num_generations_per_prompt=2
    grpo.max_num_steps=1
    grpo.max_num_epochs=1
    policy.train_global_batch_size=2
    data.train.data_path=/workspace/LexBrowserEnv/training/data/webvoyager/smoke.jsonl
    checkpointing.enabled=false
  )
fi

if [[ "$MODE" == "stage1" ]]; then
  # Stage 1 acceptance: one GRPO group of eight real WebVoyager trajectories.
  # Keep this first update deliberately at micro-batch=1: it validates that a
  # *single* 16K packed trajectory can complete the backward pass after the
  # eight rollouts have been collected.  It is not the final throughput setup;
  # the `train` mode retains micro-batch=4/global-batch=8 (two accumulation
  # steps) once this gate has passed.
  overrides=(
    grpo.num_prompts_per_step=1
    grpo.num_generations_per_prompt=8
    grpo.max_num_steps=1
    grpo.max_num_epochs=1
    policy.train_global_batch_size=8
    policy.train_micro_batch_size=1
    data.train.data_path=/workspace/LexBrowserEnv/training/data/webvoyager/smoke.jsonl
    checkpointing.enabled=false
  )
fi

if [[ "$MODE" == "stage2" ]]; then
  # Stage 2 is the throughput gate for the final configuration: one genuine
  # GRPO group (eight rollouts), consumed as two micro-batches of four before
  # one optimizer update.  Keep it to a single update so an OOM is isolated.
  overrides=(
    grpo.num_prompts_per_step=1
    grpo.num_generations_per_prompt=8
    grpo.max_num_steps=1
    grpo.max_num_epochs=1
    policy.train_global_batch_size=8
    policy.train_micro_batch_size=4
    data.train.data_path=/workspace/LexBrowserEnv/training/data/webvoyager/smoke.jsonl
    checkpointing.enabled=false
  )
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file=""
audit_container_path="/workspace/LexBrowserEnv/logs/lexbrowser-grpo/${MODE}-${timestamp}.trajectory_audit.jsonl"

# A CDP ``Page.navigate`` acknowledgement is not evidence that the remote
# Chrome can actually reach a WebVoyager website: an unavailable egress route
# later becomes chrome-error:// and used to surface as an HTTP 500/RayTaskError
# only after a full GRPO batch had been scheduled.  Gate every run on one real
# browser reachability check in the same image and virtualenv as NeMo Gym.
# The smoke helper deliberately exits non-zero on chrome-error://, ERR_*, or a
# navigation timeout, so we never train on infrastructure-failure reward=0s.
preflight_log="$ROOT/logs/lexbrowser-grpo/${MODE}-${timestamp}.preflight.log"
if [[ "$MODE" == "train" ]]; then
  # This is a backend-health gate, not a sample-level validation pass.  The
  # full 600-task corpus deliberately spans many live sites; making its first
  # row a hard gate lets one transient upstream route outage prevent all
  # training before the environment can record it as an infrastructure event.
  # Apple is a real WebVoyager origin with a verified Lexmount/CDP health
  # path.  It does not replace or filter the formal training data.
  preflight_url="${LEXBROWSER_PREFLIGHT_URL:-https://www.apple.com/}"
else
  preflight_url="$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(handle))
url = row.get("info", {}).get("start_url", "")
if not url:
    raise SystemExit("smoke task has no info.start_url")
print(url)
' "$DATA_DIR/smoke.jsonl")"
fi
set +e
"${DOCKER[@]}" run --rm \
  --network host \
  --env-file "$ROOT/secrets.env" \
  -v "$ROOT:/workspace/LexBrowserEnv" \
  -v "$GYM_CACHE_DIR/verifiers-agent:$GYM_DIR/.venv" \
  -w /workspace/LexBrowserEnv \
  "$IMAGE" \
  "$GYM_DIR/.venv/bin/python" \
  training/scripts/smoke_lexmount_cdp.py --url "$preflight_url" --timeout 45 \
  2>&1 | tee "$preflight_log"
preflight_status=${PIPESTATUS[0]}
set -e
if (( preflight_status != 0 )); then
  echo "Lexmount real-site preflight failed (status ${preflight_status}); training was not started." >&2
  echo "Inspect ${preflight_log}. Fix the Lexmount project egress route or provide the complete LEXMOUNT_EXTERNAL_PROXY_* tuple in secrets.env." >&2
  exit "$preflight_status"
fi

# vLLM's V1 engine profiles its free memory while NeMo's colocated policy
# workers are finishing their own initialization.  A worker releasing memory
# during that short window triggers vLLM's *memory profiling* assertion even
# though there is more (not less) free GPU memory.  Retry only that documented
# transient once the previous container has fully exited; do not hide OOMs,
# browser failures, or arbitrary training errors.
run_one_attempt() {
  local attempt_log="$1"
  set +e
  "${DOCKER[@]}" run --rm --name "$container_name" --gpus '"device=0,1"' \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env-file "$ROOT/secrets.env" \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e HF_HOME=/workspace/cache/huggingface \
  -e UV_CACHE_DIR=/workspace/cache/uv \
  -e LEXBROWSER_TRAJECTORY_AUDIT_LOG="$audit_container_path" \
  -e NO_PROXY=127.0.0.1,localhost \
  -e no_proxy=127.0.0.1,localhost \
  -v "$ROOT:/workspace/LexBrowserEnv" \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  -v "$WORKSPACE/.cache/nemo-rl:/workspace/cache" \
  -v "$GYM_CACHE_DIR/verifiers-agent:$GYM_DIR/.venv" \
  -v "$GYM_CACHE_DIR/vllm-model:$GYM_MODEL_DIR/.venv" \
  -v "$ROOT/training/nemo_gym/verifiers_agent_app.py:$GYM_DIR/app.py:ro" \
  -v "$ROOT/training/nemo_gym/verifiers_agent_requirements.txt:$GYM_DIR/requirements.txt:ro" \
  -v "$ROOT/training/nemo_gym/lexbrowser_webvoyager.yaml:$GYM_DIR/configs/lexbrowser_webvoyager.yaml:ro" \
  -v "$ROOT/training/nemo_rl_patches/vllm_worker.py:/opt/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker.py:ro" \
  -v "$ROOT/training/nemo_rl_patches/vllm_worker_async.py:/opt/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py:ro" \
  -v "$ROOT/training/nemo_rl_patches/grpo.py:/opt/nemo-rl/nemo_rl/algorithms/grpo.py:ro" \
  -v "$ROOT/training/configs/grpo_lexbrowser_webvoyager_qwen3_1_7b_2x5090.yaml:/workspace/config.yaml:ro" \
  -w /opt/nemo-rl \
  "$IMAGE" \
  python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config /workspace/config.yaml "${overrides[@]}" 2>&1 | tee "$attempt_log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

for attempt in 1 2 3; do
  attempt_log="$ROOT/logs/lexbrowser-grpo/${MODE}-${timestamp}.attempt${attempt}.log"
  if run_one_attempt "$attempt_log"; then
    log_file="$attempt_log"
    break
  fi
  if ! grep -q "Error in memory profiling" "$attempt_log"; then
    echo "training failed without a retryable vLLM memory-profiling race; log: $attempt_log" >&2
    exit 1
  fi
  if (( attempt == 3 )); then
    echo "vLLM memory-profiling race persisted after ${attempt} clean attempts; log: $attempt_log" >&2
    exit 1
  fi
  echo "retryable vLLM memory-profiling race; waiting for GPU cleanup before attempt $((attempt + 1))/3" >&2
  "${DOCKER[@]}" rm -f "$container_name" >/dev/null 2>&1 || true
  sleep 20
done

if [[ "$MODE" == "train" ]]; then
  "${DOCKER[@]}" run --rm \
    -v "$ROOT:/workspace/LexBrowserEnv" \
    -w /workspace/LexBrowserEnv \
    "$IMAGE" \
    python training/scripts/generate_train_report.py \
      --log-root logs/lexbrowser-grpo \
      --output-dir docs/train_reports
fi

echo "completed $MODE run; log: $log_file"
