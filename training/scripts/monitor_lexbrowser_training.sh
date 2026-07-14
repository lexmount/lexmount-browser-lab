#!/usr/bin/env bash
# Continuously append a small, secret-free training health record.  This is
# intentionally independent from the training container so a failed rollout
# still leaves an audit trail for diagnosis.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERVAL_SECONDS="${LEXBROWSER_MONITOR_INTERVAL_SECONDS:-60}"
ROLLOUTS_PER_OPTIMIZER_STEP="${LEXBROWSER_ROLLOUTS_PER_OPTIMIZER_STEP:-64}"
UNIT_PREFIX="${LEXBROWSER_UNIT_PREFIX:-lexbrowser-webvoyager}"
OUT_DIR="$ROOT/logs/lexbrowser-grpo/monitor"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$(date +%Y%m%d).jsonl"

while true; do
latest_log="$(find "$ROOT/logs/lexbrowser-grpo" -maxdepth 1 -name '*.attempt1.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
# Do not pair a fresh formal log with an audit from a previous stage run.
# The audit filename is intentionally derived from the attempt filename.
latest_audit=""
if [[ -n "$latest_log" ]]; then
  attempt_base="$(basename "$latest_log" .attempt1.log)"
  candidate_audit="$ROOT/logs/lexbrowser-grpo/${attempt_base}.trajectory_audit.jsonl"
  [[ -f "$candidate_audit" ]] && latest_audit="$candidate_audit"
fi
  python3 - "$latest_log" "$latest_audit" "$UNIT_PREFIX" "$ROLLOUTS_PER_OPTIMIZER_STEP" <<'PY' >>"$OUT"
import collections
import datetime
import json
import os
import subprocess
import sys

log_path, audit_path, unit_prefix, rollouts_per_step = sys.argv[1:]
rollouts_per_step = int(rollouts_per_step)

def run(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

units = [line.split()[0] for line in run([
    "systemctl", "list-units", "--all", "--no-legend", f"{unit_prefix}*"
]).splitlines() if line]
unit_states = {unit: run(["systemctl", "show", unit, "-p", "ActiveState", "-p", "ExecMainStatus", "--value"]).splitlines() for unit in units}
record = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "units": unit_states,
    "log": os.path.basename(log_path) if log_path else None,
    "audit": os.path.basename(audit_path) if audit_path else None,
}
if audit_path and os.path.exists(audit_path):
    rows = []
    with open(audit_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    metrics = [row.get("metrics", {}) for row in rows]
    record["rollouts_collected"] = len(rows)
    record["reward_mean"] = (sum(float(row.get("reward", 0.0)) for row in rows) / len(rows)) if rows else None
    record["valid_trajectories"] = sum(float(metric.get("valid_trajectory", 1.0)) for metric in metrics)
    valid_rows = [
        row for row in rows
        if float(row.get("metrics", {}).get("valid_trajectory", 1.0)) > 0.0
    ]
    record["reward_mean_valid_trajectory"] = (
        sum(float(row.get("reward", 0.0)) for row in valid_rows) / len(valid_rows)
        if valid_rows else None
    )
    # Formal configuration: 8 prompts x 8 sampled trajectories = 64 rows per
    # optimizer update. Browser rollouts and optimizer steps remain explicit.
    record["completed_optimizer_steps_from_audit"] = len(rows) // rollouts_per_step
    record["rollouts_per_optimizer_step"] = rollouts_per_step
    record["tool_calls"] = sum(float(metric.get("tool_calls", 0.0)) for metric in metrics)
    record["infrastructure_failures"] = sum(float(metric.get("infrastructure_failures", 0.0)) for metric in metrics)
    record["policy_failures"] = sum(float(metric.get("policy_failures", 0.0)) for metric in metrics)
    for key in (
        "rollout_wall_seconds",
        "agent_to_browser_dispatch_seconds",
        "browser_slot_wait_seconds",
        "lexmount_session_create_seconds",
        "browser_setup_navigation_seconds",
        "browser_tool_seconds",
        "browser_response_seconds",
        "policy_request_seconds",
        "judge_seconds",
    ):
        values = [float(metric.get(key, 0.0)) for metric in metrics]
        record[f"mean_{key}"] = sum(values) / len(values) if values else None
if log_path and os.path.exists(log_path):
    text = open(log_path, encoding="utf-8", errors="replace").read()
    import re
    steps = re.findall(r"Step\s+([0-9]+)/([0-9]+)", text)
    progress = re.findall(r"Collecting rollouts:\s+([0-9]+)%", text)
    losses = re.findall(r"• Loss:\s+([0-9.eE+-]+)", text)
    rewards = re.findall(r"• Avg Reward:\s+([0-9.eE+-]+)", text)
    record.update({
        "rollout_progress_percent": int(progress[-1]) if progress else None,
        "current_step": int(steps[-1][0]) if steps else None,
        "max_steps": int(steps[-1][1]) if steps else None,
        "loss": float(losses[-1]) if losses else None,
        "avg_reward_logged": float(rewards[-1]) if rewards else None,
        "oom": "out of memory" in text.lower() or "cuda oom" in text.lower(),
    })
gpu = run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
record["gpu"] = [line.strip() for line in gpu.splitlines() if line.strip()]
print(json.dumps(record, ensure_ascii=False), flush=True)
PY
  sleep "$INTERVAL_SECONDS"
done
