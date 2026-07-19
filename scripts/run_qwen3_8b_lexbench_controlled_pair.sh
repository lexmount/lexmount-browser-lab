#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCHMARK_REPO=""
ENV_FILE=""
BASE_ENV_FILE=""
TASK_FILE=""
CONFIG="$ROOT_DIR/experiments/qwen3-8b-lexbench/config.controlled-egress.yaml"
ARTIFACT_ROOT="$ROOT_DIR/artifacts/qwen3-8b-lexbench/controlled-pairs"
CONCURRENCY=1
ORDER="local-first"
LABEL="controlled-parity"

usage() {
  printf '%s\n' \
    "usage: $0 --benchmark-repo PATH --env-file PATH --task-file PATH" \
    "          [--base-env-file PATH]" \
    "          [--config PATH] [--artifact-root PATH] [--concurrency N]" \
    "          [--order local-first|lexmount-first] [--label NAME]" \
    "" \
    "Set LEXBENCH_QWEN_BASE_URL and LEXBENCH_QWEN_API_KEY to override stale" \
    "model endpoint values in --env-file for this process only."
}

while (($#)); do
  case "$1" in
    --benchmark-repo) BENCHMARK_REPO=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    --base-env-file) BASE_ENV_FILE=$2; shift 2 ;;
    --task-file) TASK_FILE=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT=$2; shift 2 ;;
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
if [[ -n "$BASE_ENV_FILE" ]]; then
  [[ -f "$BASE_ENV_FILE" ]] || { echo "invalid --base-env-file" >&2; exit 2; }
fi
[[ -f "$ENV_FILE" ]] || { echo "invalid --env-file" >&2; exit 2; }
[[ -f "$TASK_FILE" ]] || { echo "invalid --task-file" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "invalid --config" >&2; exit 2; }
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --concurrency" >&2; exit 2; }
[[ "$ORDER" == "local-first" || "$ORDER" == "lexmount-first" ]] || {
  echo "invalid --order" >&2
  exit 2
}
[[ "$LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "invalid --label" >&2; exit 2; }

set -a
if [[ -n "$BASE_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$BASE_ENV_FILE"
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
if [[ -n ${LEXBENCH_QWEN_BASE_URL:-} ]]; then
  export QWEN_BASE_URL=$LEXBENCH_QWEN_BASE_URL
fi
if [[ -n ${LEXBENCH_QWEN_API_KEY:-} ]]; then
  export QWEN_API_KEY=$LEXBENCH_QWEN_API_KEY
fi
if [[ -n ${LEXBENCH_QWEN_MODEL_ID:-} ]]; then
  export QWEN_MODEL_ID=$LEXBENCH_QWEN_MODEL_ID
fi

for name in \
  QWEN_API_KEY QWEN_BASE_URL QWEN_MODEL_ID \
  LEXBENCH_JUDGE_API_KEY LEXBENCH_JUDGE_BASE_URL LEXBENCH_JUDGE_MODEL \
  LEXMOUNT_API_KEY LEXMOUNT_PROJECT_ID \
  LEXBENCH_LOCAL_PROXY_SERVER LEXBENCH_LEXMOUNT_PROXY_SERVER \
  LEXBENCH_LEXMOUNT_PROXY_USERNAME LEXBENCH_LEXMOUNT_PROXY_PASSWORD; do
  [[ -n ${!name:-} ]] || { echo "missing required environment value: $name" >&2; exit 2; }
done
[[ "$QWEN_MODEL_ID" == "qwen3_8B" ]] || {
  echo "controlled protocol requires QWEN_MODEL_ID=qwen3_8B" >&2
  exit 2
}
[[ "$LEXBENCH_JUDGE_MODEL" == "gpt-5.4" ]] || {
  echo "controlled protocol requires LEXBENCH_JUDGE_MODEL=gpt-5.4" >&2
  exit 2
}

mapfile -t TASK_IDS < <(awk 'NF {print $1}' "$TASK_FILE")
((${#TASK_IDS[@]} > 0)) || { echo "task file is empty" >&2; exit 2; }
for task_id in "${TASK_IDS[@]}"; do
  [[ "$task_id" =~ ^[0-9]+$ ]] || { echo "invalid task id: $task_id" >&2; exit 2; }
done
[[ $(printf '%s\n' "${TASK_IDS[@]}" | sort -u | wc -l | tr -d ' ') == ${#TASK_IDS[@]} ]] || {
  echo "task file contains duplicate ids" >&2
  exit 2
}

python3 - <<'PY'
import json
import os
from urllib.request import Request, urlopen

base = os.environ["QWEN_BASE_URL"].rstrip("/")
url = base if base.endswith("/models") else f"{base}/models"
request = Request(url, headers={"Authorization": f"Bearer {os.environ['QWEN_API_KEY']}"})
with urlopen(request, timeout=15) as response:
    payload = json.load(response)
models = {str(item.get("id")) for item in payload.get("data", [])}
expected = os.environ["QWEN_MODEL_ID"]
if expected not in models:
    raise SystemExit(f"model endpoint does not serve {expected}")
print("model_endpoint=verified")
PY

if [[ "$ORDER" == "local-first" ]]; then
  BACKENDS=(local lexmount)
else
  BACKENDS=(lexmount local)
fi

DRIVER_ID="${LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
DRIVER_DIR="$ARTIFACT_ROOT/$DRIVER_ID"
mkdir -p "$DRIVER_DIR/arms"
cp "$TASK_FILE" "$DRIVER_DIR/task-ids.txt"
cp "$CONFIG" "$DRIVER_DIR/config.template.yaml"
printf '%s\n' "$ORDER" > "$DRIVER_DIR/order.txt"
printf '%s\n' "$CONCURRENCY" > "$DRIVER_DIR/concurrency.txt"
printf '%s\n' "${#TASK_IDS[@]}" > "$DRIVER_DIR/planned-task-count.txt"
printf '%s\n' "$QWEN_MODEL_ID" > "$DRIVER_DIR/model-id.txt"
printf '%s\n' "$QWEN_BASE_URL" > "$DRIVER_DIR/model-base-url.txt"
sha256sum "$TASK_FILE" > "$DRIVER_DIR/task-file.sha256"
sha256sum "$CONFIG" > "$DRIVER_DIR/config-template.sha256"

RUN_ROOT="$BENCHMARK_REPO/experiments/LexBench-Browser/All/browser-use/$QWEN_MODEL_ID"
DATASET="$BENCHMARK_REPO/browseruse_bench/data/LexBench-Browser/task.jsonl"
VLLM_INCLUDE_ARGS=()
if [[ -n ${LEXBENCH_VLLM_PID:-} ]]; then
  [[ "$LEXBENCH_VLLM_PID" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid LEXBENCH_VLLM_PID" >&2
    exit 2
  }
  VLLM_INCLUDE_ARGS=(--include-pid "$LEXBENCH_VLLM_PID")
fi

run_arm() {
  local backend=$1
  local timestamp
  timestamp=$(date -u +%Y%m%d_%H%M%S)
  local arm_dir="$DRIVER_DIR/arms/$backend"
  local run_dir="$RUN_ROOT/$timestamp"
  mkdir -p "$arm_dir"
  mkdir -p "$RUN_ROOT"
  [[ ! -e "$run_dir" ]] || {
    echo "refusing to reuse existing official run directory: $run_dir" >&2
    return 1
  }
  mkdir "$run_dir"

  local command=(
    uv run bubench run
    --agent browser-use
    --agent-config "$CONFIG"
    --data LexBench-Browser
    --split All
    --model qwen3-8B
    --browser-id "$backend"
    --timestamp "$timestamp"
    --mode specific
    --task-ids "${TASK_IDS[@]}"
    --no-group-by-site
    --concurrency "$CONCURRENCY"
  )
  if [[ "$backend" == "local" ]]; then
    command=(xvfb-run -a -s "-screen 0 1920x1080x24" "${command[@]}")
  fi

  printf 'starting %s %s\n' "$backend" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee "$arm_dir/run-started.txt"
  set +e
  uv run --project "$ROOT_DIR" python -m lexbrowser_eval.resources.cgroup_profiler \
    --output-dir "$arm_dir/resources" \
    --cwd "$BENCHMARK_REPO" \
    --label "$DRIVER_ID-$backend" \
    --planned-tasks "${#TASK_IDS[@]}" \
    "${VLLM_INCLUDE_ARGS[@]}" \
    -- "${command[@]}"
  local rollout_status=$?
  set -e
  printf '%s\n' "$rollout_status" > "$arm_dir/rollout-return-code.txt"
  [[ -d "$run_dir" ]] || {
    echo "official run directory is missing for $backend: $run_dir" >&2
    return 1
  }
  printf '%s\n' "$run_dir" > "$arm_dir/run-dir.txt"
  ((rollout_status == 0)) || {
    echo "rollout failed for $backend with status $rollout_status" >&2
    return "$rollout_status"
  }

  (
    cd "$BENCHMARK_REPO"
    uv run bubench eval \
      --agent browser-use \
      --agent-config "$CONFIG" \
      --data LexBench-Browser \
      --split All \
      --model-id "$QWEN_MODEL_ID" \
      --timestamp "$timestamp" \
      --num-worker 5 \
      --eval-strategy stepwise
  ) 2>&1 | tee "$arm_dir/eval.log"

  uv run --project "$ROOT_DIR" python -m lexbrowser_eval.lexbench.summarize \
    --run-dir "$run_dir" \
    --dataset "$DATASET" \
    --resource-summary "$arm_dir/resources/resource_summary.json" \
    --output "$arm_dir/benchmark_summary.json"
  printf '%s\t%s\n' "$backend" "$run_dir" >> "$DRIVER_DIR/runs.tsv"
}

for backend in "${BACKENDS[@]}"; do
  run_arm "$backend"
  sleep 1
done

LEXMOUNT_ARM="$DRIVER_DIR/arms/lexmount"
LOCAL_ARM="$DRIVER_DIR/arms/local"
LEXMOUNT_RUN=$(<"$LEXMOUNT_ARM/run-dir.txt")
LOCAL_RUN=$(<"$LOCAL_ARM/run-dir.txt")
LEXMOUNT_EVAL=$(find "$LEXMOUNT_RUN/tasks_eval_result" -maxdepth 1 -name '*_eval_results.json' | sort | tail -1)
LOCAL_EVAL=$(find "$LOCAL_RUN/tasks_eval_result" -maxdepth 1 -name '*_eval_results.json' | sort | tail -1)
[[ -n "$LEXMOUNT_EVAL" && -n "$LOCAL_EVAL" ]] || {
  echo "missing official evaluation output" >&2
  exit 1
}

uv run --project "$ROOT_DIR" python -m lexbrowser_eval.lexbench.compare_pair \
  --lexmount "$LEXMOUNT_ARM/benchmark_summary.json" \
  --local "$LOCAL_ARM/benchmark_summary.json" \
  --output "$DRIVER_DIR/pair_comparison.json"
uv run --project "$ROOT_DIR" python -m lexbrowser_eval.lexbench.audit_paired_runs \
  --dataset "$DATASET" \
  --lexmount-summary "$LEXMOUNT_ARM/benchmark_summary.json" \
  --local-summary "$LOCAL_ARM/benchmark_summary.json" \
  --lexmount-run "$LEXMOUNT_RUN" \
  --local-run "$LOCAL_RUN" \
  --lexmount-eval "$LEXMOUNT_EVAL" \
  --local-eval "$LOCAL_EVAL" \
  --output "$DRIVER_DIR/pair_audit.json"

printf '%s\n' "$DRIVER_DIR"
