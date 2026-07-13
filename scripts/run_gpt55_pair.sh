#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCHMARK_REPO=${BENCHMARK_REPO:-}
ENV_FILE=${ENV_FILE:-}
TASK_FILE=""
CONCURRENCY=""
ORDER=""
LABEL=""

usage() {
  printf '%s\n' \
    "usage: $0 --benchmark-repo PATH --env-file PATH --task-file PATH" \
    "          --concurrency N --order local-first|lexmount-first --label NAME"
}

while (($#)); do
  case "$1" in
    --benchmark-repo) BENCHMARK_REPO=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    --task-file) TASK_FILE=$2; shift 2 ;;
    --concurrency) CONCURRENCY=$2; shift 2 ;;
    --order) ORDER=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$BENCHMARK_REPO/.git" || -f "$BENCHMARK_REPO/.git" ]] || {
  echo "invalid --benchmark-repo" >&2
  exit 2
}
[[ -f "$ENV_FILE" ]] || { echo "invalid --env-file" >&2; exit 2; }
[[ -f "$TASK_FILE" ]] || { echo "invalid --task-file" >&2; exit 2; }
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --concurrency" >&2; exit 2; }
[[ "$ORDER" == "local-first" || "$ORDER" == "lexmount-first" ]] || {
  echo "invalid --order" >&2
  exit 2
}
[[ "$LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "invalid --label" >&2; exit 2; }

if [[ "$ORDER" == "local-first" ]]; then
  BACKENDS=(local lexmount)
else
  BACKENDS=(lexmount local)
fi

DRIVER_ID="${LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
DRIVER_DIR="$ROOT_DIR/artifacts/gpt55-lexbench/drivers/$DRIVER_ID"
mkdir -p "$DRIVER_DIR"
printf '%s\n' "$ORDER" > "$DRIVER_DIR/order.txt"
printf '%s\n' "$CONCURRENCY" > "$DRIVER_DIR/concurrency.txt"
cp "$TASK_FILE" "$DRIVER_DIR/task-ids.txt"

status=0
for backend in "${BACKENDS[@]}"; do
  backend_log="$DRIVER_DIR/$backend.log"
  printf 'starting %s at %s\n' "$backend" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$backend_log"
  if "$ROOT_DIR/scripts/run_gpt55_lexbench.sh" \
    --benchmark-repo "$BENCHMARK_REPO" \
    --env-file "$ENV_FILE" \
    --backend "$backend" \
    --phase audit \
    --concurrency "$CONCURRENCY" \
    --task-file "$TASK_FILE" 2>&1 | tee -a "$backend_log"; then
    artifact_path=$(tail -n 1 "$backend_log")
    printf '%s\t%s\n' "$backend" "$artifact_path" >> "$DRIVER_DIR/runs.tsv"
  else
    status=$?
    printf '%s\t%s\n' "$backend" "$status" >> "$DRIVER_DIR/failures.tsv"
    break
  fi
done

printf '%s\n' "$status" > "$DRIVER_DIR/status.txt"
printf '%s\n' "$DRIVER_DIR"
exit "$status"
