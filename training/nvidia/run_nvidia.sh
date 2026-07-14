#!/usr/bin/env bash
# Submit or enter a reproducible Lexmount WebVoyager NVIDIA delivery run.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELIVERY_DIR="$ROOT/training/nvidia"
MODE="train"
NODES="8"
GPUS_PER_NODE="8"
GPU_FAMILY="${LEXBROWSER_GPU_FAMILY:-H100}"
RUN_ROOT="${LEXBROWSER_RUN_ROOT:-$ROOT/artifacts/lexbrowser-nvidia}"
RUN_ID=""
RESUME_FROM=""
WAIT=1
MODEL_ID="${LEXBROWSER_MODEL_ID:-Qwen/Qwen3-1.7B}"
# Pin the upstream model snapshot rather than allowing a moving `main` branch
# to change a later rerun. This is the Qwen/Qwen3-1.7B commit resolved on 2026-07-14.
MODEL_REVISION="${LEXBROWSER_MODEL_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
IMAGE="${NEMO_RL_IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"

usage() {
  cat <<'EOF'
Usage: training/nvidia/run_nvidia.sh [options]

  --mode MODE             dry-run | smoke | node-check | train (default: train)
  --nodes N               requested Slurm nodes (default: 8)
  --gpus-per-node N       requested GPUs on each node (default: 8)
  --gpu-family NAME       expected GPU family (default: H100; use any to skip family matching)
  --run-id ID             stable output directory name (default: UTC timestamp)
  --run-root PATH         shared artifact root
  --resume PATH           resume only this explicit NeMo checkpoint directory
  --no-wait               submit and return the Slurm job id immediately
  --help                  show this message

Environment overrides: NEMO_RL_IMAGE, LEXBROWSER_SLURM_PARTITION,
LEXBROWSER_SLURM_ACCOUNT, LEXBROWSER_SLURM_TIME, LEXBROWSER_SLURM_GRES,
LEXBROWSER_MODEL_ID, LEXBROWSER_MODEL_REVISION, LEXBROWSER_RUN_ROOT, and
LEXBROWSER_GPU_FAMILY.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --gpu-family) GPU_FAMILY="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --resume) RESUME_FROM="$2"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in dry-run|smoke|node-check|train) ;; *) echo "unsupported mode: $MODE" >&2; exit 2 ;; esac
for number in "$NODES" "$GPUS_PER_NODE"; do
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo "expected positive integer, got: $number" >&2; exit 2; }
done
[[ "$GPU_FAMILY" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid GPU family: $GPU_FAMILY" >&2; exit 2; }
[[ -n "$RUN_ID" ]] || RUN_ID="${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid run id: $RUN_ID" >&2; exit 2; }
[[ -z "$RESUME_FROM" || "$RESUME_FROM" =~ ^/[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || {
  echo "resume path must be an absolute path using only letters, digits, '.', '_', '-', and '/'" >&2
  exit 2
}

RUN_DIR="$RUN_ROOT/$RUN_ID"
CONFIG="$DELIVERY_DIR/configs/grpo_lexbrowser_webvoyager_qwen3_1_7b_nvidia.yaml"
CONTRACT="$DELIVERY_DIR/configs/comparison_contract.json"
DATASET="$ROOT/training/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/datasets/WebVoyager_data_clean.jsonl"
SECRETS="$ROOT/secrets.env"

mkdir -p "$RUN_DIR"
init_args=(
  --run-dir "$RUN_DIR" --root "$ROOT" --config "$CONFIG" --dataset "$DATASET"
  --secrets-file "$SECRETS" --mode "$MODE" --backend lexmount --nodes "$NODES"
  --gpus-per-node "$GPUS_PER_NODE" --gpu-family "$GPU_FAMILY"
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION"
  --comparison-contract "$(tr -d '\n' < "$CONTRACT")"
)
[[ -n "$RESUME_FROM" ]] && init_args+=(--resume-from "$RESUME_FROM")
python3 "$DELIVERY_DIR/scripts/run_manifest.py" init "${init_args[@]}"

if [[ "$MODE" == "dry-run" ]]; then
  python3 "$DELIVERY_DIR/scripts/run_manifest.py" phase --run-dir "$RUN_DIR" --name submission --status complete --detail "dry-run only; no Slurm allocation or network request"
  python3 "$DELIVERY_DIR/scripts/run_manifest.py" finalize --run-dir "$RUN_DIR" --status complete --detail "dry-run complete"
  printf 'dry-run manifest: %s/manifests/run_manifest.json\n' "$RUN_DIR"
  exit 0
fi

if [[ ! -f "$SECRETS" ]]; then
  echo "missing $SECRETS; copy secrets.env.example and keep it private" >&2
  exit 1
fi
if stat -c '%a' "$SECRETS" >/dev/null 2>&1; then
  secrets_mode="$(stat -c '%a' "$SECRETS")"
else
  secrets_mode="$(stat -f '%Lp' "$SECRETS")"
fi
if [[ "$secrets_mode" != "600" ]]; then
  echo "secrets.env must have file mode 600" >&2
  exit 1
fi

# Do not let a caller's exported credentials reach preflight containers. The
# entrypoint re-sources this private file only after node, NCCL, and Gym gates.
unset LEXMOUNT_BASE_URL LEXMOUNT_API_KEY LEXMOUNT_PROJECT_ID LEXMOUNT_REGION
unset LEXMOUNT_EXTERNAL_PROXY_SERVER LEXMOUNT_EXTERNAL_PROXY_USERNAME
unset LEXMOUNT_EXTERNAL_PROXY_PASSWORD OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL HF_TOKEN

# Some managed platforms invoke this launcher from an existing Slurm allocation
# instead of exposing a submission host. Reuse that allocation rather than
# nesting an sbatch request, while retaining the exact same entrypoint.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export LEXBROWSER_MODE="$MODE" LEXBROWSER_RUN_DIR="$RUN_DIR" LEXBROWSER_ROOT="$ROOT"
  export LEXBROWSER_NODES="$NODES" LEXBROWSER_GPUS_PER_NODE="$GPUS_PER_NODE"
  export LEXBROWSER_GPU_FAMILY="$GPU_FAMILY"
  export LEXBROWSER_MODEL_ID="$MODEL_ID" LEXBROWSER_MODEL_REVISION="$MODEL_REVISION"
  export LEXBROWSER_NEMO_IMAGE="$IMAGE" LEXBROWSER_RESUME_FROM="$RESUME_FROM"
  python3 "$DELIVERY_DIR/scripts/run_manifest.py" phase --run-dir "$RUN_DIR" --name submission --status started --detail "using existing Slurm allocation $SLURM_JOB_ID"
  exec "$DELIVERY_DIR/scripts/slurm_entrypoint.sh"
fi
command -v sbatch >/dev/null || { echo "sbatch is required outside an allocation" >&2; exit 1; }

export_vars="PATH=$PATH,HOME=${HOME:-},USER=${USER:-},LEXBROWSER_MODE=$MODE,LEXBROWSER_RUN_DIR=$RUN_DIR,LEXBROWSER_ROOT=$ROOT,LEXBROWSER_NODES=$NODES,LEXBROWSER_GPUS_PER_NODE=$GPUS_PER_NODE,LEXBROWSER_GPU_FAMILY=$GPU_FAMILY,LEXBROWSER_MODEL_ID=$MODEL_ID,LEXBROWSER_MODEL_REVISION=$MODEL_REVISION,LEXBROWSER_NEMO_IMAGE=$IMAGE,LEXBROWSER_RESUME_FROM=$RESUME_FROM,LEXBROWSER_MIN_FREE_MIB=${LEXBROWSER_MIN_FREE_MIB:-},LEXBROWSER_NCCL_MASTER_PORT=${LEXBROWSER_NCCL_MASTER_PORT:-},LEXBROWSER_METRICS_INTERVAL_SECONDS=${LEXBROWSER_METRICS_INTERVAL_SECONDS:-},LEXBROWSER_PREFLIGHT_URL=${LEXBROWSER_PREFLIGHT_URL:-},NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-},NCCL_IB_HCA=${NCCL_IB_HCA:-},NCCL_DEBUG=${NCCL_DEBUG:-}"
sbatch_args=(--nodes="$NODES" --ntasks-per-node=1 --exclusive --job-name="lexbrowser-${MODE}" --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" --export="$export_vars")
[[ -n "${LEXBROWSER_SLURM_PARTITION:-}" ]] && sbatch_args+=(--partition="$LEXBROWSER_SLURM_PARTITION")
[[ -n "${LEXBROWSER_SLURM_ACCOUNT:-}" ]] && sbatch_args+=(--account="$LEXBROWSER_SLURM_ACCOUNT")
[[ -n "${LEXBROWSER_SLURM_TIME:-}" ]] && sbatch_args+=(--time="$LEXBROWSER_SLURM_TIME")
if [[ -n "${LEXBROWSER_SLURM_GRES:-}" ]]; then
  sbatch_args+=(--gres="$LEXBROWSER_SLURM_GRES")
else
  sbatch_args+=(--gres="gpu:$GPUS_PER_NODE")
fi
[[ "$WAIT" -eq 1 ]] && sbatch_args+=(--wait)

python3 "$DELIVERY_DIR/scripts/run_manifest.py" phase --run-dir "$RUN_DIR" --name submission --status started --detail "submitting ${NODES}x${GPUS_PER_NODE} ${GPU_FAMILY} allocation"
set +e
submission="$(sbatch "${sbatch_args[@]}" "$DELIVERY_DIR/scripts/slurm_entrypoint.sh" 2>&1)"
status=$?
set -e
printf '%s\n' "$submission"
if [[ "$status" -ne 0 ]]; then
  python3 "$DELIVERY_DIR/scripts/run_manifest.py" finalize --run-dir "$RUN_DIR" --status failed --detail "sbatch submission failed"
  exit "$status"
fi
printf 'run directory: %s\n' "$RUN_DIR"
