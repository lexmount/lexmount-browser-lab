#!/usr/bin/env bash
# Start the Ray head or a Ray worker container on one H100 host.
#
# CUDA port of the internal Ascend per-node script: NVIDIA GPUs replace the
# /dev/davinci* Ascend devices, and NCCL_* replaces the HCCL_* transport
# configuration. Everything else (mounts, verl patches, env plumbing) is
# unchanged.
set -Eeuo pipefail

ROLE=${ROLE:?set ROLE=head or ROLE=worker}
NODE_IP=${NODE_IP:?set NODE_IP}
HEAD_IP=${HEAD_IP:?set HEAD_IP}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$SCRIPT_DIR}
RUNS_ROOT=${RUNS_ROOT:-/data/lexbrowser-rl/runs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/data/lexbrowser-rl/checkpoints}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}
IMAGE=${IMAGE:-lexbrowser-verl-h100:local}
NAME=${NAME:-lexbrowser-h100-ray}
SHM_SIZE=${SHM_SIZE:-128g}
VERL_AGENT_LOOP_PATCH=${VERL_AGENT_LOOP_PATCH:-$ROOT/runtime/patches/agent_loop.py}
VERL_DISTRIBUTED_PATCH=${VERL_DISTRIBUTED_PATCH:-$ROOT/runtime/patches/distributed.py}
LEXBROWSER_ACTION_MAX_TOKENS=${LEXBROWSER_ACTION_MAX_TOKENS:-1024}
VERL_PROCESS_GROUP_TIMEOUT_SECONDS=${VERL_PROCESS_GROUP_TIMEOUT_SECONDS:-7200}
SECRETS_FILE=${SECRETS_FILE:-$ROOT/secrets.env}

for patch in "$VERL_AGENT_LOOP_PATCH" "$VERL_DISTRIBUTED_PATCH"; do
  if [[ ! -f "$patch" ]]; then
    echo "Missing persistent verl patch: $patch" >&2
    exit 1
  fi
done
if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing secrets file: $SECRETS_FILE (copy training/h100/secrets.env.example)" >&2
  exit 1
fi
if ! [[ "$VERL_PROCESS_GROUP_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] ||
   (( VERL_PROCESS_GROUP_TIMEOUT_SECONDS < 1800 )); then
  echo "VERL_PROCESS_GROUP_TIMEOUT_SECONDS must be an integer >= 1800." >&2
  exit 2
fi
for path in "$ROOT" "$RUNS_ROOT" "$CHECKPOINT_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "Required shared path is missing: $path" >&2
    exit 1
  fi
done

# Verl uses Gloo for its cross-node process-group coordination. Without an
# explicit interface, the container hostname can resolve to loopback, which
# makes ranks on different hosts try to dial 127.0.0.1. Bind both Gloo and
# NCCL's socket transport to the NIC that carries NODE_IP.
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(
  ip -o -4 addr show | awk -v node_ip="$NODE_IP" '$4 ~ ("^" node_ip "/") { print $2; exit }'
)}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME%%@*}
if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
  echo "Could not find the network interface for NODE_IP=$NODE_IP" >&2
  exit 1
fi
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}

extra_env=()
[[ -n "${NEMO_GYM_BROWSER_URL:-}" ]] && extra_env+=(-e "NEMO_GYM_BROWSER_URL=$NEMO_GYM_BROWSER_URL")
[[ -n "${LEXBROWSER_METRICS_DIR:-}" ]] && extra_env+=(-e "LEXBROWSER_METRICS_DIR=$LEXBROWSER_METRICS_DIR")
# Optional InfiniBand tuning knobs; pass through only when the operator set
# them (bare TCP fabrics should leave them unset).
[[ -n "${NCCL_IB_HCA:-}" ]] && extra_env+=(-e "NCCL_IB_HCA=$NCCL_IB_HCA")
[[ -n "${NCCL_IB_GID_INDEX:-}" ]] && extra_env+=(-e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX")
[[ -n "${NCCL_DEBUG:-}" ]] && extra_env+=(-e "NCCL_DEBUG=$NCCL_DEBUG")

docker rm -f "$NAME" >/dev/null 2>&1 || true
if [[ "$ROLE" == head ]]; then
  ray_cmd="ray start --head --node-ip-address=${NODE_IP} --port=6379 --dashboard-host=0.0.0.0 --block"
else
  ray_cmd="ray start --address=${HEAD_IP}:6379 --node-ip-address=${NODE_IP} --block"
fi

docker run -d --name "$NAME" --network host --ipc host --shm-size "$SHM_SIZE" \
  --gpus all \
  -v "$ROOT:/workspace/lexbrowser-h100:ro" \
  -v "$RUNS_ROOT:/workspace/runs" \
  -v "$CHECKPOINT_ROOT:/workspace/checkpoints" \
  -v "$VERL_AGENT_LOOP_PATCH:/verl/verl/experimental/agent_loop/agent_loop.py:ro" \
  -v "$VERL_DISTRIBUTED_PATCH:/verl/verl/utils/distributed.py:ro" \
  -v "$MODEL_PATH:$MODEL_PATH:ro" \
  --env-file "$SECRETS_FILE" \
  -e PYTHONPATH=/workspace/lexbrowser-h100/runtime:/workspace/lexbrowser-h100/runtime/lexbrowser_webvoyager/src \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e LEXBROWSER_ACTION_MAX_TOKENS="$LEXBROWSER_ACTION_MAX_TOKENS" \
  -e VERL_PROCESS_GROUP_TIMEOUT_SECONDS="$VERL_PROCESS_GROUP_TIMEOUT_SECONDS" \
  "${extra_env[@]}" \
  "$IMAGE" bash -lc "$ray_cmd"
