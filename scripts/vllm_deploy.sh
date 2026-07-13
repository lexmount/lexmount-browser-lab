#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/home/wf/.venv-qwen3-vllm}"
MODEL_DIR="${MODEL_DIR:-/home/wf/models/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18088}"
MODEL_NAME="${MODEL_NAME:-qwen3_8B}"
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-abc123}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.48}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing venv: $VENV_DIR" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Missing model directory: $MODEL_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

exec vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$OPENAI_API_KEY" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --reasoning-parser qwen3
