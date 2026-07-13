#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=""
args=()

while (($#)); do
  case "$1" in
    --env-file)
      (($# >= 2)) || { echo "--env-file requires a path" >&2; exit 2; }
      env_file=$2
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
usage: run_online_mind2web.sh --env-file PATH [options]

options:
  --backend lexmount|local|all
  --stage prepare|rollout|judge|report|all
  --campaign-id ID
  --task-count N (N<300 validates the full blob then runs the fixed first_n slice)
  --runtime-root PATH
  --max-rollout-passes N
  --allow-partial-judge
EOF
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

[[ -n "$env_file" && -r "$env_file" ]] || {
  echo "missing readable --env-file" >&2
  exit 64
}

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

export JUDGE_API_KEY=${JUDGE_API_KEY:-${LEXBENCH_JUDGE_API_KEY:-}}
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-${LEXBENCH_JUDGE_BASE_URL:-}}

exec uv run --project "$ROOT_DIR" \
  python -m lexbrowser_eval.online_mind2web.cli "${args[@]}"
