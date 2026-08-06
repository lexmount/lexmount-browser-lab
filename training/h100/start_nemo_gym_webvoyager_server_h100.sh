#!/usr/bin/env bash
# Start the CPU-only NeMo-Gym browser environment sidecar on the Ray head.
#
# Browser-environment sidecar launcher (CUDA image and neutral paths;
# the server itself, vendored at runtime/nemo_gym_webvoyager_server.py, is
# hardware-neutral).
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$SCRIPT_DIR}
WORK_ROOT=${WORK_ROOT:-/data/lexbrowser-rl}
RUNS_ROOT=${RUNS_ROOT:-$WORK_ROOT/runs}
NEMO_GYM_ROOT=${NEMO_GYM_ROOT:-$WORK_ROOT/cache/nemo-gym-v0.2.1}
RUNTIME_SITE=${RUNTIME_SITE:-$WORK_ROOT/runtime/nemo-gym-site}
IMAGE=${IMAGE:-lexbrowser-verl-h100:local}
NAME=${NAME:-lexbrowser-nemo-gym-webvoyager}
PORT=${PORT:-18180}
MAX_CONCURRENT_SESSIONS=${MAX_CONCURRENT_SESSIONS:-64}
MAX_CONCURRENT_CREATES=${MAX_CONCURRENT_CREATES:-16}
SESSION_CREATE_ATTEMPTS=${LEXMOUNT_SESSION_CREATE_ATTEMPTS:-2}
RESET_TOTAL_BUDGET_S=${LEXMOUNT_RESET_TOTAL_BUDGET_S:-110}
SESSION_CREATE_TIMEOUT_S=${LEXMOUNT_SESSION_CREATE_TIMEOUT_S:-60}
RESET_CIRCUIT_COOLDOWN_S=${LEXMOUNT_RESET_CIRCUIT_COOLDOWN_S:-60}
SWEEP_INTERVAL_S=${LEXMOUNT_SWEEP_INTERVAL_S:-60}
JUDGE_REQUEST_TIMEOUT_S=${LEXBROWSER_JUDGE_REQUEST_TIMEOUT_S:-45}
EXECUTOR_WORKERS=${LEXBROWSER_EXECUTOR_WORKERS:-$((MAX_CONCURRENT_SESSIONS + MAX_CONCURRENT_CREATES + 16))}
BROWSER_BACKEND=${BROWSER_BACKEND:-lexmount}
# Provider browser engine: "normal" is the Chrome-based cloud browser, "light"
# selects lightmount (chrome-light-docker), the in-house engine — the same
# switch the SDK quickstart makes in light_demo.py.
BROWSER_MODE=${LEXMOUNT_BROWSER_MODE:-normal}
LOCAL_CDP_HTTP_URL=${LOCAL_CDP_HTTP_URL:-http://127.0.0.1:9222}
JUDGE_TRANSCRIPT_CHAR_LIMIT=${LEXBROWSER_JUDGE_TRANSCRIPT_CHAR_LIMIT:-60000}
JUDGE_MAX_ATTEMPTS=${LEXBROWSER_JUDGE_MAX_ATTEMPTS:-3}
# Judges tried in order when the primary refuses the prompt on content policy.
# The primary endpoint moderated WebVoyager's BBC News tasks on 2026-08-05
# (HTTP 400, vendor code 10013); a refusal is a property of the prompt, so the
# only way out is a different judge, not another attempt.
JUDGE_FALLBACK_MODELS=${LEXBROWSER_JUDGE_FALLBACK_MODELS:-}
AUDIT_DIR=${AUDIT_DIR:-$RUNS_ROOT/manual/audit}
SECRETS_FILE=${SECRETS_FILE:-$ROOT/secrets.env}

for path in "$ROOT" "$NEMO_GYM_ROOT/nemo_gym" "$RUNTIME_SITE" "$SECRETS_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 1
  fi
done
mkdir -p "$AUDIT_DIR"

if docker inspect "$NAME" >/dev/null 2>&1; then
  docker stop --time 300 "$NAME" >/dev/null 2>&1 || true
  docker rm "$NAME" >/dev/null 2>&1 || true
fi
docker run -d --name "$NAME" --network host --ipc host \
  --env-file "$SECRETS_FILE" \
  -e BROWSER_BACKEND="$BROWSER_BACKEND" \
  -e LEXMOUNT_BROWSER_MODE="$BROWSER_MODE" \
  -e LOCAL_CDP_HTTP_URL="$LOCAL_CDP_HTTP_URL" \
  -e LEXMOUNT_MAX_CONCURRENT_SESSIONS="$MAX_CONCURRENT_SESSIONS" \
  -e LEXMOUNT_MAX_CONCURRENT_CREATES="$MAX_CONCURRENT_CREATES" \
  -e LEXMOUNT_SESSION_CREATE_ATTEMPTS="$SESSION_CREATE_ATTEMPTS" \
  -e LEXMOUNT_RESET_TOTAL_BUDGET_S="$RESET_TOTAL_BUDGET_S" \
  -e LEXMOUNT_SESSION_CREATE_TIMEOUT_S="$SESSION_CREATE_TIMEOUT_S" \
  -e LEXMOUNT_RESET_CIRCUIT_COOLDOWN_S="$RESET_CIRCUIT_COOLDOWN_S" \
  -e LEXMOUNT_SWEEP_INTERVAL_S="$SWEEP_INTERVAL_S" \
  -e LEXBROWSER_JUDGE_REQUEST_TIMEOUT_S="$JUDGE_REQUEST_TIMEOUT_S" \
  -e LEXBROWSER_EXECUTOR_WORKERS="$EXECUTOR_WORKERS" \
  -e LEXBROWSER_JUDGE_TRANSCRIPT_CHAR_LIMIT="$JUDGE_TRANSCRIPT_CHAR_LIMIT" \
  -e LEXBROWSER_JUDGE_MAX_ATTEMPTS="$JUDGE_MAX_ATTEMPTS" \
  -e LEXBROWSER_JUDGE_FALLBACK_MODELS="$JUDGE_FALLBACK_MODELS" \
  -e LEXBROWSER_AUDIT_DIR=/audit \
  -e PYTHONPATH=/runtime:/opt/nemo-gym:/workspace/lexbrowser-h100/runtime:/workspace/lexbrowser-h100/runtime/lexbrowser_webvoyager/src \
  -v "$ROOT:/workspace/lexbrowser-h100:ro" \
  -v "$AUDIT_DIR:/audit" \
  -v "$NEMO_GYM_ROOT:/opt/nemo-gym:ro" \
  -v "$RUNTIME_SITE:/runtime:ro" \
  -w /workspace/lexbrowser-h100 \
  --entrypoint python3 \
  "$IMAGE" runtime/nemo_gym_webvoyager_server.py --port "$PORT" >/dev/null

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
    echo "NEMO_GYM_WEBVOYAGER_SERVER_OK url=http://127.0.0.1:$PORT backend=$BROWSER_BACKEND browser_mode=$BROWSER_MODE"
    exit 0
  fi
  if ! docker inspect "$NAME" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
    docker logs "$NAME" >&2 || true
    exit 1
  fi
  sleep 1
done

docker logs "$NAME" >&2 || true
echo "NeMo-Gym WebVoyager server did not become healthy" >&2
exit 1
