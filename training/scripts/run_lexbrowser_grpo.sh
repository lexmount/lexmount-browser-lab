#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(dirname "$ROOT")"
IMAGE="${NEMO_RL_IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"
MODEL_DIR="${MODEL_DIR:-/home/wf/models/Qwen3-1.7B}"
MODE="${1:-train}"
DATA_SOURCE="$ROOT/training/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/datasets/WebVoyager_data_clean.jsonl"
DATA_DIR="$ROOT/training/data/webvoyager"
GYM_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/verifiers_agent"
GYM_MODEL_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model"
GYM_CACHE_DIR="$WORKSPACE/.cache/nemo-rl/gym-venvs"

if [[ "$MODE" != "train" && "$MODE" != "smoke" ]]; then
  echo "usage: $0 [train|smoke]" >&2
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
  "$WORKSPACE/.cache/nemo-rl" "$GYM_CACHE_DIR/verifiers-agent" "$GYM_CACHE_DIR/vllm-model"
python3 "$ROOT/training/scripts/prepare_webvoyager_data.py" \
  --source "$DATA_SOURCE" \
  --output "$DATA_DIR/train.jsonl" \
  --manifest "$DATA_DIR/manifest.json"
python3 "$ROOT/training/scripts/prepare_webvoyager_data.py" \
  --source "$DATA_SOURCE" \
  --output "$DATA_DIR/smoke.jsonl" \
  --manifest "$DATA_DIR/smoke-manifest.json" \
  --limit 1

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

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="$ROOT/logs/lexbrowser-grpo/${MODE}-${timestamp}.log"

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
  -v "$ROOT/training/configs/grpo_lexbrowser_webvoyager_qwen3_1_7b_2x5090.yaml:/workspace/config.yaml:ro" \
  -w /opt/nemo-rl \
  "$IMAGE" \
  python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config /workspace/config.yaml "${overrides[@]}" 2>&1 | tee "$log_file"

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
