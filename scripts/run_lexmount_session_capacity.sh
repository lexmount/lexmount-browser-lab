#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=""
OUTPUT_DIR=""
SMOKE_COUNT=4
FULL_EN_COUNT=32
FULL_ZH_COUNT=32

usage() {
  printf '%s\n' \
    "usage: $0 --env-file PATH --output-dir PATH" \
    "          [--smoke-count N] [--full-en-count N] [--full-zh-count N]"
}

while (($#)); do
  case "$1" in
    --env-file) ENV_FILE=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --smoke-count) SMOKE_COUNT=$2; shift 2 ;;
    --full-en-count) FULL_EN_COUNT=$2; shift 2 ;;
    --full-zh-count) FULL_ZH_COUNT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "invalid --env-file" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "missing --output-dir" >&2; exit 2; }
for value in "$SMOKE_COUNT" "$FULL_EN_COUNT" "$FULL_ZH_COUNT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "session counts must be positive" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"

run_probe() {
  local label=$1
  local en_count=$2
  local zh_count=$3
  local hold_seconds=$4
  uv run --project "$ROOT_DIR" python -m \
    lexbrowser_eval.lexbench.probe_multi_profile_sessions \
    --env-file "$ENV_FILE" \
    --en-count "$en_count" \
    --zh-count "$zh_count" \
    --hold-seconds "$hold_seconds" \
    --poll-timeout-seconds 180 \
    --sample-interval-seconds 1 \
    --cleanup-grace-seconds 120 \
    --cleanup-poll-seconds 5 \
    --output "$OUTPUT_DIR/$label.json" \
    | tee "$OUTPUT_DIR/$label.log"
}

run_probe smoke "$SMOKE_COUNT" "$SMOKE_COUNT" 10
run_probe full "$FULL_EN_COUNT" "$FULL_ZH_COUNT" 60

printf '%s\n' "$OUTPUT_DIR"
