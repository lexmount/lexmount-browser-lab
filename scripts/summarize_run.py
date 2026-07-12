#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

TIMEOUT_RE = re.compile(r"timed?\s*out|timeout", re.IGNORECASE)
SESSION_RE = re.compile(
    r"session.{0,40}(quota|creat.{0,12}fail)|insufficient.*session", re.IGNORECASE
)
NETWORK_RE = re.compile(
    r"ERR_(?:NETWORK_CHANGED|CONNECTION|TIMED_OUT)|navigation failed|net::",
    re.IGNORECASE,
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def describe(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = value if isinstance(value, list) else [value]
    return [record for record in records if isinstance(record, dict)]


def _load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    records = _load_json_records(path)
    return {
        str(record.get("id") or record.get("task_id")): record
        for record in records
        if record.get("id") not in (None, "") or record.get("task_id") not in (None, "")
    }


def _planned_ids(run_dir: Path, dataset: dict[str, dict[str, Any]]) -> list[str]:
    snapshot = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    context = snapshot.get("run") or {}
    mode = context.get("mode")
    if mode == "all":
        return list(dataset)
    if mode == "specific":
        return [str(value) for value in context.get("task_ids") or []]
    if mode == "by_id":
        value = context.get("task_id")
        return [str(value)] if value not in (None, "") else []
    return [path.parent.name for path in sorted((run_dir / "tasks").glob("*/result.json"))]


def _eval_records(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "tasks_eval_result").glob("*_eval_results.json"))
    return _load_json_records(paths[-1]) if paths else []


def _round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return value


def summarize_run(
    run_dir: Path, dataset_path: Path, resource_summary_path: Path | None = None
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    planned_ids = _planned_ids(run_dir, dataset)
    result_paths = sorted((run_dir / "tasks").glob("*/result.json"))
    results = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in result_paths
    }
    eval_records = _eval_records(run_dir)
    judged = {str(record.get("task_id")): record for record in eval_records}

    per_task: dict[str, dict[str, Any]] = {}
    token_total = 0
    agent_cost_total = 0.0
    judge_token_total = 0
    steps: list[float] = []
    e2e_seconds: list[float] = []
    error_counts = defaultdict(int)
    strata: dict[str, dict[str, int]] = defaultdict(
        lambda: {"planned": 0, "trajectory": 0, "judged": 0, "success": 0}
    )

    for task_id in planned_ids:
        metadata = dataset.get(task_id, {})
        region = str(metadata.get("website_region") or "unknown")
        task_type = str(metadata.get("task_type") or "unknown")
        stratum = f"{region}/{task_type}"
        strata[stratum]["planned"] += 1
        result = results.get(task_id)
        evaluation = judged.get(task_id)
        success = bool(evaluation and int(evaluation.get("predicted_label") or 0) == 1)
        task_summary: dict[str, Any] = {
            "region": region,
            "task_type": task_type,
            "trajectory": result is not None,
            "judged": evaluation is not None,
            "success": success,
        }
        if result is not None:
            strata[stratum]["trajectory"] += 1
            metrics = result.get("metrics") or {}
            usage = metrics.get("usage") or {}
            task_steps = metrics.get("steps")
            task_e2e_ms = metrics.get("end_to_end_ms")
            task_wall = result.get("wall_clock_seconds")
            task_tokens = int(usage.get("total_tokens") or 0)
            task_cost = float(usage.get("total_cost") or 0.0)
            token_total += task_tokens
            agent_cost_total += task_cost
            if task_steps is not None:
                steps.append(float(task_steps))
            if task_e2e_ms is not None:
                e2e_seconds.append(float(task_e2e_ms) / 1000)
            elif task_wall is not None:
                e2e_seconds.append(float(task_wall))
            serialized = json.dumps(
                {
                    "error": result.get("error"),
                    "env_status": result.get("env_status"),
                    "action_history": result.get("action_history"),
                },
                ensure_ascii=False,
            )
            signals = {
                "timeout": bool(TIMEOUT_RE.search(serialized)),
                "session_create": bool(SESSION_RE.search(serialized)),
                "network_navigation": bool(NETWORK_RE.search(serialized)),
                "unhandled_error": bool(result.get("error")),
            }
            for key, present in signals.items():
                error_counts[key] += int(present)
            task_summary.update(
                {
                    "agent_done": result.get("agent_done"),
                    "agent_success": result.get("agent_success"),
                    "env_status": result.get("env_status"),
                    "steps": task_steps,
                    "e2e_seconds": float(task_e2e_ms) / 1000 if task_e2e_ms else task_wall,
                    "tokens": task_tokens,
                    "agent_cost": task_cost,
                    "signals": signals,
                }
            )
        if evaluation is not None:
            strata[stratum]["judged"] += 1
            strata[stratum]["success"] += int(success)
            details = evaluation.get("evaluation_details") or {}
            task_summary["judge_score"] = details.get("score")
            task_summary["judge_tokens"] = (details.get("eval_usage") or {}).get("total_tokens")
            judge_token_total += int(task_summary["judge_tokens"] or 0)
        per_task[task_id] = task_summary

    planned = len(planned_ids)
    trajectory_count = len(results)
    judged_count = len(judged)
    success_count = sum(item["success"] for item in per_task.values())
    agent_done_count = sum(item.get("agent_done") == "done" for item in per_task.values())
    agent_success_count = sum(bool(item.get("agent_success")) for item in per_task.values())
    duration = sum(e2e_seconds)
    resource_summary = None
    throughput = None
    if resource_summary_path and resource_summary_path.exists():
        resource_summary = json.loads(resource_summary_path.read_text(encoding="utf-8"))
        rollout_duration = float(resource_summary.get("duration_seconds") or 0)
        if rollout_duration > 0:
            throughput = trajectory_count / rollout_duration * 3600
    return _round_nested(
        {
            "schema_version": 1,
            "run_dir": str(run_dir.resolve()),
            "dataset": str(dataset_path.resolve()),
            "counts": {
                "planned": planned,
                "trajectory": trajectory_count,
                "judged": judged_count,
                "success": success_count,
                "agent_done": agent_done_count,
                "agent_success": agent_success_count,
            },
            "rates": {
                "trajectory_per_planned": trajectory_count / planned if planned else None,
                "success_per_planned": success_count / planned if planned else None,
                "success_per_judged": success_count / judged_count if judged_count else None,
            },
            "steps": describe(steps),
            "e2e_seconds": describe(e2e_seconds),
            "agent_usage": {
                "total_tokens": token_total,
                "total_cost": agent_cost_total,
            },
            "judge_usage": {"total_tokens": judge_token_total},
            "error_task_counts": dict(error_counts),
            "strata": dict(sorted(strata.items())),
            "sum_task_e2e_seconds": duration,
            "throughput_task_per_hour": throughput,
            "resource_summary": resource_summary,
            "per_task": per_task,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one browseruse-agent-bench run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resource-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_run(args.run_dir, args.dataset, args.resource_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
