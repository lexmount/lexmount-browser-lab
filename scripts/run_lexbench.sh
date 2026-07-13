#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
usage:
  run_lexbench.sh qwen3-8b --env-file PATH [qwen options]
  run_lexbench.sh gpt5.5 --benchmark-repo PATH --env-file PATH [gpt5.5 options]

qwen options:
  --backend lexmount|local|all
  --mode quality|stress|all
  --stage prepare|rollout|judge|report|all
  --campaign-id ID
  --task-count N (N<210 runs an isolated official first_n smoke slice)
  --runtime-root PATH
  --resume
  --dry-run

gpt5.5 options:
  --backend lexmount|local
  --phase smoke|pilot|full|capacity
  --concurrency N
  --count N
  --skip-eval
EOF
}

(($#)) || { usage >&2; exit 2; }
experiment=$1
shift

case "$experiment" in
  qwen3-8b)
    exec uv run --project "$ROOT_DIR" \
      python -m lexbrowser_eval.lexbench.cli qwen3-8b "$@"
    ;;
  gpt5.5)
    exec "$ROOT_DIR/scripts/run_gpt55_lexbench.sh" "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    printf 'unknown LexBench experiment: %s\n' "$experiment" >&2
    usage >&2
    exit 2
    ;;
esac
