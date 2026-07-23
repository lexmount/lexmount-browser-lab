#!/usr/bin/env bash
# Cluster preflight for the H100 recipe. Run from the Ray head.
#
# Per-node checks: paths, secrets, model, image, disk, GPU count, CUDA probe,
# NeMo-Gym import, and a real Lexmount browser session smoke.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$SCRIPT_DIR}
WORK_ROOT=${WORK_ROOT:-/data/lexbrowser-rl}
RUNS_ROOT=${RUNS_ROOT:-$WORK_ROOT/runs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$WORK_ROOT/checkpoints}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}
IMAGE=${IMAGE:-lexbrowser-verl-h100:local}
NEMO_GYM_ROOT=${NEMO_GYM_ROOT:-$WORK_ROOT/cache/nemo-gym-v0.2.1}
RUNTIME_SITE=${RUNTIME_SITE:-$WORK_ROOT/runtime/nemo-gym-site}
NODES_CSV=${NODES_CSV:?set NODES_CSV, e.g. NODES_CSV=<head-ip> or NODES_CSV=<head-ip>,<worker-ip>}
HEAD_IP=${HEAD_IP:-${NODES_CSV%%,*}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MIN_FREE_GB=${MIN_FREE_GB:-200}
STORAGE_PATH=${STORAGE_PATH:-$CHECKPOINT_ROOT}
SMOKE_URL=${SMOKE_URL:-https://arxiv.org/}
BROWSER_BACKEND=${BROWSER_BACKEND:-lexmount}
LOCAL_CDP_HTTP_URL=${LOCAL_CDP_HTTP_URL:-http://127.0.0.1:9222}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_USER=${SSH_USER:-root}
SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
IFS=',' read -r -a NODES <<<"$NODES_CSV"

check_node() {
  local node=$1
  local script
  read -r -d '' script <<'NODE_SCRIPT' || true
set -Eeuo pipefail
test -d "$ROOT"
mkdir -p "$RUNS_ROOT" "$CHECKPOINT_ROOT"
test -d "$RUNS_ROOT"
test -d "$CHECKPOINT_ROOT"
test -f "$ROOT/secrets.env"
test -f "$MODEL_PATH/config.json"
test -d "$NEMO_GYM_ROOT/nemo_gym"
test -d "$RUNTIME_SITE"
docker image inspect "$IMAGE" >/dev/null
run_probe="$RUNS_ROOT/.write-probe-$(hostname)-$$"
checkpoint_probe="$CHECKPOINT_ROOT/.write-probe-$(hostname)-$$"
touch "$run_probe" "$checkpoint_probe"
rm -f "$run_probe" "$checkpoint_probe"
free_gb=$(df -Pk "$STORAGE_PATH" | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= MIN_FREE_GB ))
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$gpu_count" == "$GPUS_PER_NODE" ]]
docker run --rm --gpus all --entrypoint python3 "$IMAGE" -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA_OK", torch.cuda.device_count(), torch.version.cuda)' >/dev/null
docker run --rm --network host \
  -e PYTHONPATH=/runtime:/opt/nemo-gym \
  -v "$RUNTIME_SITE:/runtime:ro" \
  -v "$NEMO_GYM_ROOT:/opt/nemo-gym:ro" \
  --entrypoint python3 "$IMAGE" -c \
  'from nemo_gym.base_resources_server import SimpleResourcesServer' >/dev/null
case "$BROWSER_BACKEND" in
  lexmount)
    curl -fsS --connect-timeout 10 --max-time 20 "${LEXMOUNT_CONSOLE_URL:-https://browser.lexmount.com/}" >/dev/null
    docker run --rm --network host --env-file "$ROOT/secrets.env" \
      -v "$ROOT:/workspace/lexbrowser-h100:ro" \
      -w /workspace/lexbrowser-h100 --entrypoint python3 "$IMAGE" \
      runtime/smoke_lexmount_cdp.py --url "$SMOKE_URL" --timeout 60
    ;;
  local_cdp)
    if [[ "$NODE_IP" == "$HEAD_IP" ]]; then
      curl -fsS --connect-timeout 3 --max-time 5 "$LOCAL_CDP_HTTP_URL/json/version" \
        | grep -q webSocketDebuggerUrl
    fi
    ;;
  *)
    echo "Unsupported BROWSER_BACKEND=$BROWSER_BACKEND" >&2
    exit 2
    ;;
esac
printf 'NODE_PREFLIGHT_OK host=%s free_gb=%s gpu_count=%s backend=%s\n' \
  "$(hostname)" "$free_gb" "$gpu_count" "$BROWSER_BACKEND"
NODE_SCRIPT

  if [[ "$node" == "${NODES[0]}" ]]; then
    env ROOT="$ROOT" RUNS_ROOT="$RUNS_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" MODEL_PATH="$MODEL_PATH" IMAGE="$IMAGE" \
      NEMO_GYM_ROOT="$NEMO_GYM_ROOT" RUNTIME_SITE="$RUNTIME_SITE" \
      MIN_FREE_GB="$MIN_FREE_GB" STORAGE_PATH="$STORAGE_PATH" GPUS_PER_NODE="$GPUS_PER_NODE" \
      SMOKE_URL="$SMOKE_URL" BROWSER_BACKEND="$BROWSER_BACKEND" \
      LOCAL_CDP_HTTP_URL="$LOCAL_CDP_HTTP_URL" NODE_IP="$node" HEAD_IP="$HEAD_IP" \
      bash -lc "$script"
  else
    printf '%s\n' "$script" | "${SSH[@]}" "$SSH_USER@$node" env ROOT="$ROOT" RUNS_ROOT="$RUNS_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" MODEL_PATH="$MODEL_PATH" \
      IMAGE="$IMAGE" NEMO_GYM_ROOT="$NEMO_GYM_ROOT" RUNTIME_SITE="$RUNTIME_SITE" \
      MIN_FREE_GB="$MIN_FREE_GB" STORAGE_PATH="$STORAGE_PATH" GPUS_PER_NODE="$GPUS_PER_NODE" \
      SMOKE_URL="$SMOKE_URL" BROWSER_BACKEND="$BROWSER_BACKEND" \
      LOCAL_CDP_HTTP_URL="$LOCAL_CDP_HTTP_URL" NODE_IP="$node" HEAD_IP="$HEAD_IP" \
      bash -s
  fi
}

for node in "${NODES[@]}"; do
  echo "PREFLIGHT_NODE_START node=$node"
  check_node "$node"
done

echo "CLUSTER_PREFLIGHT_OK nodes=${#NODES[@]}"
