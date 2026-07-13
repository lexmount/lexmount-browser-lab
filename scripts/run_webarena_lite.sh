#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=""
backend="playwright"
args=()

while (($#)); do
  case "$1" in
    --env-file)
      (($# >= 2)) || { echo "--env-file requires a path" >&2; exit 2; }
      env_file=$2
      shift 2
      ;;
    --backend)
      (($# >= 2)) || { echo "--backend requires a value" >&2; exit 2; }
      backend=$2
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
usage: run_webarena_lite.sh --env-file PATH --backend playwright [options]

options:
  --runtime-root PATH
  --server HOST
  --map-server HOST
  --work-dir PATH
  --result-dir PATH
  --start N
  --end N
  --smoke
  --score-only
  --skip-bootstrap
  --skip-install
  --skip-prepare
  --allow-unhealthy-sites

Only the official Playwright backend is implemented in the current runner.
EOF
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

[[ "$backend" == "playwright" ]] || {
  printf 'unsupported WebArena-Lite backend: %s; expected playwright\n' "$backend" >&2
  exit 2
}
[[ -n "$env_file" && -r "$env_file" ]] || {
  echo "missing readable --env-file" >&2
  exit 64
}

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

exec uv run --project "$ROOT_DIR" \
  python -m lexbrowser_eval.webarena_lite.cli "${args[@]}"
