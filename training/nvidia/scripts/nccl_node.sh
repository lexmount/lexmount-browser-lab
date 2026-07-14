#!/usr/bin/env bash
set -Eeuo pipefail

: "${LEXBROWSER_NODES:?}" "${LEXBROWSER_GPUS_PER_NODE:?}" "${LEXBROWSER_CONTAINER_RUN_DIR:?}"
head_node="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
master_port="${LEXBROWSER_NCCL_MASTER_PORT:-29573}"
exec torchrun \
  --nnodes="$LEXBROWSER_NODES" \
  --nproc-per-node="$LEXBROWSER_GPUS_PER_NODE" \
  --node-rank="$SLURM_PROCID" \
  --master-addr="$head_node" \
  --master-port="$master_port" \
  /workspace/LexBrowserEnv/training/nvidia/scripts/nccl_smoke.py
