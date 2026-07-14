#!/usr/bin/env bash
# Runs inside the Slurm allocation; credentials never appear in scheduler args.
set -Eeuo pipefail

: "${LEXBROWSER_MODE:?}" "${LEXBROWSER_RUN_DIR:?}" "${LEXBROWSER_ROOT:?}"
: "${LEXBROWSER_NODES:?}" "${LEXBROWSER_GPUS_PER_NODE:?}" "${LEXBROWSER_NEMO_IMAGE:?}"

DELIVERY_DIR="$LEXBROWSER_ROOT/training/nvidia"
RUN_DIR="$LEXBROWSER_RUN_DIR"
CONTAINER_RUN_DIR="/workspace/LexBrowserEnv${RUN_DIR#"$LEXBROWSER_ROOT"}"
EXPECTED_GPUS="$LEXBROWSER_GPUS_PER_NODE"
EXPECTED_GPU_FAMILY="${LEXBROWSER_GPU_FAMILY:?LEXBROWSER_GPU_FAMILY is required}"
MIN_FREE_MIB="${LEXBROWSER_MIN_FREE_MIB:-70000}"
GYM_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/verifiers_agent"
VLLM_GYM_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model"
SHARED_CACHE="$LEXBROWSER_ROOT/.cache/nemo-rl-nvidia"

phase() { python3 "$DELIVERY_DIR/scripts/run_manifest.py" phase --run-dir "$RUN_DIR" --name "$1" --status "$2" --detail "$3"; }
fail() {
  phase "${1:-run}" failed "${2:-unknown failure}" || true
  python3 "$DELIVERY_DIR/scripts/run_manifest.py" finalize --run-dir "$RUN_DIR" --status failed --detail "${2:-unknown failure}" || true
  exit 1
}
trap 'fail allocation "Slurm entrypoint failed"' ERR

[[ "${SLURM_JOB_NUM_NODES:-}" == "$LEXBROWSER_NODES" ]] || fail allocation "Slurm allocated ${SLURM_JOB_NUM_NODES:-0} nodes; expected $LEXBROWSER_NODES"
command -v srun >/dev/null || fail allocation "srun unavailable"
srun --help 2>&1 | grep -q -- '--container-image' || fail allocation "Slurm Pyxis/Enroot support (--container-image) is required"
mkdir -p "$RUN_DIR" "$SHARED_CACHE/gym-venvs/verifiers-agent" "$SHARED_CACHE/gym-venvs/vllm-model" "$RUN_DIR/cache/huggingface" "$RUN_DIR/cache/uv" "$RUN_DIR/logs" "$RUN_DIR/metrics" "$RUN_DIR/preflight/nodes"

resume_env_value="$(printf '%q' "${LEXBROWSER_RESUME_FROM:-}")"
cat > "$RUN_DIR/run.env" <<EOF
export LEXBROWSER_MODE='$LEXBROWSER_MODE'
export LEXBROWSER_RUN_DIR='$CONTAINER_RUN_DIR'
export LEXBROWSER_HOST_RUN_DIR='$RUN_DIR'
export LEXBROWSER_ROOT='/workspace/LexBrowserEnv'
export LEXBROWSER_NODES='$LEXBROWSER_NODES'
export LEXBROWSER_GPUS_PER_NODE='$LEXBROWSER_GPUS_PER_NODE'
export LEXBROWSER_GPU_FAMILY='$EXPECTED_GPU_FAMILY'
export LEXBROWSER_MODEL_ID='$LEXBROWSER_MODEL_ID'
export LEXBROWSER_MODEL_REVISION='$LEXBROWSER_MODEL_REVISION'
export LEXBROWSER_RESUME_FROM=$resume_env_value
export HF_HOME='$CONTAINER_RUN_DIR/cache/huggingface'
export UV_CACHE_DIR='$CONTAINER_RUN_DIR/cache/uv'
export PYTHONPATH='/workspace/LexBrowserEnv/training/nemo_gym/runtime_overrides'
export LEXBROWSER_STABLE_GRPO_GROUPING=1
export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
EOF

MOUNTS="$LEXBROWSER_ROOT:/workspace/LexBrowserEnv,$SHARED_CACHE/gym-venvs/verifiers-agent:$GYM_DIR/.venv,$SHARED_CACHE/gym-venvs/vllm-model:$VLLM_GYM_DIR/.venv,$LEXBROWSER_ROOT/training/nemo_gym/verifiers_agent_app.py:$GYM_DIR/app.py:ro,$LEXBROWSER_ROOT/training/nemo_gym/verifiers_agent_requirements.txt:$GYM_DIR/requirements.txt:ro,$LEXBROWSER_ROOT/training/nemo_gym/lexbrowser_webvoyager.yaml:$GYM_DIR/configs/lexbrowser_webvoyager.yaml:ro"
CONTAINER_ARGS=(--no-container-mount-home --container-image="$LEXBROWSER_NEMO_IMAGE" --container-mounts="$MOUNTS" --container-workdir=/workspace/LexBrowserEnv)

phase preflight started "collecting one hardware report per allocated node"
export LEXBROWSER_EXPECTED_GPUS="$EXPECTED_GPUS" LEXBROWSER_EXPECTED_GPU_FAMILY="$EXPECTED_GPU_FAMILY" LEXBROWSER_MIN_FREE_MIB="$MIN_FREE_MIB"
srun "${CONTAINER_ARGS[@]}" --nodes="$LEXBROWSER_NODES" --ntasks="$LEXBROWSER_NODES" --ntasks-per-node=1 bash /workspace/LexBrowserEnv/training/nvidia/scripts/preflight_node.sh
python3 "$DELIVERY_DIR/scripts/validate_preflight.py" --nodes-dir "$RUN_DIR/preflight/nodes" --output "$RUN_DIR/preflight/summary.json" --expected-nodes "$LEXBROWSER_NODES" --expected-gpus-per-node "$LEXBROWSER_GPUS_PER_NODE" --expected-gpu-family "$EXPECTED_GPU_FAMILY"
phase preflight complete "all allocated nodes passed ${EXPECTED_GPU_FAMILY}, GPU-memory, and shared-output checks"

if [[ "$LEXBROWSER_MODE" == "node-check" ]]; then
  python3 "$DELIVERY_DIR/scripts/run_manifest.py" finalize --run-dir "$RUN_DIR" --status complete --detail "node-check complete; no credentials or training launched"
  exit 0
fi

phase agent_environment started "creating the shared NeMo Gym environment"
srun "${CONTAINER_ARGS[@]}" --nodes=1 --ntasks=1 bash /workspace/LexBrowserEnv/training/nvidia/scripts/bootstrap_agent_env.sh
srun "${CONTAINER_ARGS[@]}" --nodes="$LEXBROWSER_NODES" --ntasks="$LEXBROWSER_NODES" --ntasks-per-node=1 bash -lc "'$GYM_DIR/.venv/bin/python' -c 'import lexmount, stagehand, verifiers; print(\"agent env ok\")'"
phase agent_environment complete "NeMo Gym dependencies import on every allocated node"

phase nccl_smoke started "running a ${LEXBROWSER_NODES}x${LEXBROWSER_GPUS_PER_NODE} NCCL all-reduce"
export LEXBROWSER_CONTAINER_RUN_DIR="$CONTAINER_RUN_DIR"
srun "${CONTAINER_ARGS[@]}" --nodes="$LEXBROWSER_NODES" --ntasks="$LEXBROWSER_NODES" --ntasks-per-node=1 bash /workspace/LexBrowserEnv/training/nvidia/scripts/nccl_node.sh
phase nccl_smoke complete "NCCL all-reduce completed across requested topology"

SECRETS="$LEXBROWSER_ROOT/secrets.env"
[[ -f "$SECRETS" ]] || fail credentials "secrets.env disappeared after submission"
# shellcheck disable=SC1090
set +x
source "$SECRETS"
for key in LEXMOUNT_BASE_URL LEXMOUNT_API_KEY LEXMOUNT_PROJECT_ID OPENAI_API_KEY OPENAI_BASE_URL; do
  [[ -n "${!key:-}" ]] || fail credentials "required secret name is empty: $key"
done

phase ray_cluster started "starting NeMo RL v0.6 Ray cluster"
export CONTAINER="$LEXBROWSER_NEMO_IMAGE" MOUNTS="$MOUNTS" GPUS_PER_NODE="$LEXBROWSER_GPUS_PER_NODE"
export BASE_LOG_DIR="$RUN_DIR/logs" RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"
export COMMAND="bash /workspace/LexBrowserEnv/training/nvidia/scripts/driver.sh --run-dir '$CONTAINER_RUN_DIR'"
bash "$DELIVERY_DIR/vendor/nemo-rl-v0.6.0-ray.sub"
