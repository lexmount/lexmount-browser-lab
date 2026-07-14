#!/usr/bin/env bash
# Build the shared verifier environment once, before Ray schedules any rollout.
set -Eeuo pipefail

GYM_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/verifiers_agent"
VENV="$GYM_DIR/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  command -v uv >/dev/null || { echo "uv is required in the NeMo RL image" >&2; exit 1; }
  cd "$GYM_DIR"
  uv venv "$VENV"
  uv pip install --python "$VENV/bin/python" -r requirements.txt
fi
"$VENV/bin/python" - <<'PY'
import importlib.metadata
import json
import lexmount
import stagehand
import verifiers

print(json.dumps({
    "lexmount": importlib.metadata.version("lexmount"),
    "stagehand": importlib.metadata.version("stagehand"),
    "verifiers": importlib.metadata.version("verifiers"),
}, sort_keys=True))
PY
