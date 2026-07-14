#!/usr/bin/env bash
# Probe the fourteen real WebVoyager origins through exactly the same
# Lexmount credentials and CDP transport used by the training environment.
# This is a preflight diagnostic only: it never starts vLLM, Ray, or GRPO.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-official}"
TIMEOUT="${LEXBROWSER_SITE_PROBE_TIMEOUT:-12}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/training/.venv/bin/python}"
DATASET="$ROOT/training/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/datasets/WebVoyager_data_clean.jsonl"

if [[ "$MODE" != "official" && "$MODE" != "direct" ]]; then
  echo "usage: $0 [official|direct]" >&2
  exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python runtime: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$DATASET" && -f "$ROOT/secrets.env" ]] || { echo "missing dataset or secrets.env" >&2; exit 2; }

set -a
# shellcheck disable=SC1091
source "$ROOT/secrets.env"
set +a

mapfile -t origins < <("$PYTHON_BIN" - "$DATASET" <<'PY'
import json
import sys

seen = set()
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    name, url = row["web_name"], row["web"]
    if name not in seen:
        seen.add(name)
        print(f"{name}\t{url}")
PY
)

printf 'WEBVOYAGER_SITE_PROBE mode=%s timeout_s=%s sites=%s\n' "$MODE" "$TIMEOUT" "${#origins[@]}"
for item in "${origins[@]}"; do
  site="${item%%$'\t'*}"
  url="${item#*$'\t'}"
  output=""
  status=0
  set +e
  if [[ "$MODE" == "official" ]]; then
    output="$("$PYTHON_BIN" "$ROOT/training/scripts/smoke_lexmount_cdp.py" \
      --official-proxy --url "$url" --timeout "$TIMEOUT" 2>&1)"
    status=$?
  else
    output="$("$PYTHON_BIN" "$ROOT/training/scripts/smoke_lexmount_cdp.py" \
      --url "$url" --timeout "$TIMEOUT" 2>&1)"
    status=$?
  fi
  set -e
  # The final line is intentionally one machine-readable result per site.
  printf 'SITE=%s URL=%s status=%s %s\n' "$site" "$url" "$status" "$(tail -n 1 <<<"$output")"
done
