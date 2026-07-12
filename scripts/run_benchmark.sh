#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT_DIR/experiments/gpt55-lexbench/config.yaml"
BENCHMARK_REPO=${BENCHMARK_REPO:-}
ENV_FILE=${ENV_FILE:-}
BACKEND=""
PHASE=""
CONCURRENCY=""
COUNT=""
SKIP_EVAL=0

usage() {
  printf '%s\n' \
    "usage: $0 --benchmark-repo PATH --env-file PATH --backend lexmount|local" \
    "          --phase smoke|pilot|full|capacity --concurrency N [--count N] [--skip-eval]"
}

while (($#)); do
  case "$1" in
    --benchmark-repo) BENCHMARK_REPO=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    --phase) PHASE=$2; shift 2 ;;
    --concurrency) CONCURRENCY=$2; shift 2 ;;
    --count) COUNT=$2; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$BENCHMARK_REPO/.git" || -f "$BENCHMARK_REPO/.git" ]] || { echo "invalid --benchmark-repo" >&2; exit 2; }
[[ -f "$ENV_FILE" ]] || { echo "invalid --env-file" >&2; exit 2; }
[[ "$BACKEND" == "lexmount" || "$BACKEND" == "local" ]] || { echo "invalid --backend" >&2; exit 2; }
[[ "$PHASE" =~ ^(smoke|pilot|full|capacity)$ ]] || { echo "invalid --phase" >&2; exit 2; }
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --concurrency" >&2; exit 2; }

EXPECTED_SHA=b9d3dc655aec3cd10fd1fb86cfb7678bb9a27399
ACTUAL_SHA=$(git -C "$BENCHMARK_REPO" rev-parse HEAD)
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || {
  printf 'benchmark commit mismatch: expected %s, got %s\n' "$EXPECTED_SHA" "$ACTUAL_SHA" >&2
  exit 2
}

# browseruse-agent-bench still uses its root config to enable the agent registry,
# while --agent-config controls the resolved model/browser runtime. A dedicated
# clean worktree may safely point both reads at this experiment's fixed config.
ROOT_CONFIG="$BENCHMARK_REPO/config.yaml"
if [[ -e "$ROOT_CONFIG" || -L "$ROOT_CONFIG" ]]; then
  [[ "$(realpath "$ROOT_CONFIG")" == "$(realpath "$CONFIG")" ]] || {
    echo "benchmark config.yaml already exists and is not this experiment config" >&2
    exit 2
  }
else
  ln -s "$CONFIG" "$ROOT_CONFIG"
fi

TASK_ARGS=()
PLANNED_TASKS=""
case "$PHASE" in
  smoke)
    mapfile -t TASK_IDS < "$ROOT_DIR/experiments/gpt55-lexbench/task_sets/smoke.txt"
    TASK_ARGS=(--mode specific --task-ids "${TASK_IDS[@]}")
    PLANNED_TASKS=${#TASK_IDS[@]}
    ;;
  pilot)
    mapfile -t TASK_IDS < "$ROOT_DIR/experiments/gpt55-lexbench/task_sets/pilot20.txt"
    TASK_ARGS=(--mode specific --task-ids "${TASK_IDS[@]}")
    PLANNED_TASKS=${#TASK_IDS[@]}
    ;;
  full)
    TASK_ARGS=(--mode all)
    PLANNED_TASKS=210
    ;;
  capacity)
    [[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "capacity requires --count" >&2; exit 2; }
    CAPACITY_IDS="$ROOT_DIR/artifacts/task_sets/capacity-${COUNT}.txt"
    uv run --project "$ROOT_DIR" python "$ROOT_DIR/scripts/select_tasks.py" \
      --dataset "$BENCHMARK_REPO/browseruse_bench/data/LexBench-Browser/task.jsonl" \
      --count "$COUNT" --output "$CAPACITY_IDS"
    mapfile -t TASK_IDS < "$CAPACITY_IDS"
    TASK_ARGS=(--mode specific --task-ids "${TASK_IDS[@]}")
    PLANNED_TASKS=${#TASK_IDS[@]}
    ;;
esac

RUN_ID="${PHASE}-${BACKEND}-c${CONCURRENCY}-$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$ROOT_DIR/artifacts/gpt55-lexbench/$RUN_ID"
MARKER="$OUTPUT_DIR/benchmark-output-dir.txt"
mkdir -p "$OUTPUT_DIR"

uv run --project "$ROOT_DIR" python "$ROOT_DIR/scripts/profile_command.py" \
  --output-dir "$OUTPUT_DIR" \
  --cwd "$BENCHMARK_REPO" \
  --label "$RUN_ID" \
  --planned-tasks "$PLANNED_TASKS" \
  -- \
  uv run --env-file "$ENV_FILE" scripts/run.py \
    --agent browser-use \
    --agent-config "$CONFIG" \
    --data LexBench-Browser \
    --split All \
    --model gpt-5.5 \
    --browser "$BACKEND" \
    --concurrency "$CONCURRENCY" \
    --machine-id 5090-local-comparison \
    --write-output-dir "$MARKER" \
    "${TASK_ARGS[@]}"

[[ -s "$MARKER" ]] || { echo "benchmark output marker missing" >&2; exit 1; }
BENCHMARK_OUTPUT=$(<"$MARKER")
TIMESTAMP=$(basename "$BENCHMARK_OUTPUT")

if ((SKIP_EVAL == 0)); then
  (
    cd "$BENCHMARK_REPO"
    uv run --env-file "$ENV_FILE" scripts/eval.py \
      --agent browser-use \
      --agent-config "$CONFIG" \
      --data LexBench-Browser \
      --model-id gpt-5.5 \
      --timestamp "$TIMESTAMP" \
      --num-worker 5 \
      --eval-strategy stepwise
  ) | tee "$OUTPUT_DIR/eval.log"
fi

uv run --project "$ROOT_DIR" python "$ROOT_DIR/scripts/summarize_run.py" \
  --run-dir "$BENCHMARK_OUTPUT" \
  --dataset "$BENCHMARK_REPO/browseruse_bench/data/LexBench-Browser/task.jsonl" \
  --resource-summary "$OUTPUT_DIR/resource_summary.json" \
  --output "$OUTPUT_DIR/benchmark_summary.json"

printf '%s\n' "$OUTPUT_DIR"
