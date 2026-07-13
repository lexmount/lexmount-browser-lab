#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .compare_pair import compare_pair


def _run_id(summary: dict[str, Any]) -> str:
    return Path(str(summary.get("run_dir") or "unknown")).name


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    resource = summary.get("resource_summary") or {}
    return {
        "run_id": _run_id(summary),
        "counts": summary.get("counts"),
        "rates": summary.get("rates"),
        "steps": summary.get("steps"),
        "e2e_seconds": summary.get("e2e_seconds"),
        "throughput_task_per_hour": summary.get("throughput_task_per_hour"),
        "error_task_counts": summary.get("error_task_counts"),
        "agent_usage": summary.get("agent_usage"),
        "judge_usage": summary.get("judge_usage"),
        "resource": {
            "duration_seconds": resource.get("duration_seconds"),
            "return_code": resource.get("return_code"),
            "guard_triggered": resource.get("guard_triggered"),
            "metrics": resource.get("metrics"),
        },
    }


def compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": probe["profile"],
        "requested": probe["requested"],
        "created_within_timeout": probe["created"],
        "failed_or_timed_out": probe["failed"],
        "poll_timeout_seconds": probe["poll_timeout_seconds"],
        "create_seconds": probe["create_seconds"],
        "sessions_before_active": (probe.get("sessions_before") or {}).get("active"),
    }


def analyze_local_rerun(
    lexmount_full: dict[str, Any],
    local_full: dict[str, Any],
    local_smoke: dict[str, Any],
    local_rerun: dict[str, Any],
    paired_audit: dict[str, Any] | None = None,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    original_local = local_full["per_task"]
    rerun_tasks = local_rerun["per_task"]
    original_failures = {task_id for task_id, task in original_local.items() if not task["success"]}
    unexpected = sorted(set(rerun_tasks) - original_failures)
    if unexpected:
        raise ValueError(f"rerun contains original local successes: {unexpected}")

    recovered = sorted(
        (task_id for task_id, task in rerun_tasks.items() if task["success"]),
        key=lambda value: int(value),
    )
    adjusted_local = copy.deepcopy(local_full)
    for task_id in recovered:
        adjusted_local["per_task"][task_id]["success"] = True

    sensitivity = compare_pair(
        lexmount_full,
        adjusted_local,
        bootstrap_samples=bootstrap_samples,
    )
    recovered_states = {
        "lexmount_only": [
            task_id for task_id in recovered if lexmount_full["per_task"][task_id]["success"]
        ],
        "both_failed": [
            task_id for task_id in recovered if not lexmount_full["per_task"][task_id]["success"]
        ],
    }

    smoke_tasks = local_smoke["per_task"]
    overlap = sorted(set(smoke_tasks) & set(rerun_tasks), key=lambda value: int(value))
    smoke_success = {task_id for task_id in overlap if smoke_tasks[task_id]["success"]}
    rerun_success = {task_id for task_id in overlap if rerun_tasks[task_id]["success"]}

    result = {
        "original_local_failures": len(original_failures),
        "rerun": _compact_summary(local_rerun),
        "recovered_count": len(recovered),
        "recovered_task_ids": recovered,
        "recovered_from_original_state": recovered_states,
        "smoke_repeatability": {
            "tasks": len(overlap),
            "smoke_success_task_ids": sorted(smoke_success, key=lambda value: int(value)),
            "full_rerun_success_task_ids": sorted(rerun_success, key=lambda value: int(value)),
            "success_in_both_task_ids": sorted(
                smoke_success & rerun_success, key=lambda value: int(value)
            ),
            "flipped_task_ids": sorted(smoke_success ^ rerun_success, key=lambda value: int(value)),
        },
        "local_favoring_sensitivity": {
            "method": (
                "keep every original local success and replace only original failures "
                "that succeeded in the 5090 rerun"
            ),
            "comparison": sensitivity,
        },
    }
    if paired_audit is not None:
        environment_categories = {"E1", "E2", "E3"}
        original_environment_losers = [
            item
            for item in paired_audit["discordant"]
            if item["outcome"] == "lexmount_only"
            and item["local"].get("failure_category") in environment_categories
        ]
        environment_recovered = sorted(
            (
                item["task_id"]
                for item in original_environment_losers
                if item["task_id"] in recovered
            ),
            key=lambda value: int(value),
        )
        result["original_environment_loser_followup"] = {
            "tasks": len(original_environment_losers),
            "recovered": len(environment_recovered),
            "still_failed": len(original_environment_losers) - len(environment_recovered),
            "recovered_task_ids": environment_recovered,
            "source": "original Judge primary category E1/E2/E3",
        }
    return result


def analyze_capacity(
    c10: dict[str, Any],
    c64: dict[str, Any],
    c10_sessions: dict[str, Any] | None = None,
    c64_sessions: dict[str, Any] | None = None,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    c10_ids = set(c10["per_task"])
    c64_ids = set(c64["per_task"])
    if c10_ids != c64_ids:
        raise ValueError("capacity runs do not contain the same task ids")

    comparison = compare_pair(c64, c10, bootstrap_samples=bootstrap_samples)
    comparison["success"] = {
        "c64": comparison["success"].pop("lexmount"),
        "c10": comparison["success"].pop("local"),
    }
    comparison["paired_table"] = {
        "both_success": comparison["paired_table"]["both_success"],
        "c64_only": comparison["paired_table"]["lexmount_only"],
        "c10_only": comparison["paired_table"]["local_only"],
        "both_failed": comparison["paired_table"]["both_failed"],
    }
    difference = comparison.pop("success_rate_difference")
    comparison["success_rate_difference"] = {
        "c64_minus_c10": difference["lexmount_minus_local"],
        "bootstrap_95_ci": difference["bootstrap_95_ci"],
        "bootstrap_samples": difference["bootstrap_samples"],
    }
    comparison.pop("noninferiority")

    arms = {"c10": _compact_summary(c10), "c64": _compact_summary(c64)}
    sustainable: dict[str, bool] = {}
    for name, summary in (("c10", c10), ("c64", c64)):
        counts = summary["counts"]
        resources = summary.get("resource_summary") or {}
        sustainable[name] = (
            counts["trajectory"] == counts["planned"]
            and int(summary.get("error_task_counts", {}).get("session_create", 0)) == 0
            and resources.get("guard_triggered") is None
        )

    result = {
        "same_task_set": True,
        "task_count": len(c10_ids),
        "arms": arms,
        "paired_quality": comparison,
        "sustainable_before_external_session_check": sustainable,
    }
    session_monitors = {"c10": c10_sessions, "c64": c64_sessions}
    available_monitors = {
        name: {
            "sample_count": monitor["sample_count"],
            "duration_seconds": monitor["duration_seconds"],
            "active_sessions": monitor["active_sessions"],
            "errors": len(monitor.get("errors", [])),
        }
        for name, monitor in session_monitors.items()
        if monitor is not None
    }
    if available_monitors:
        result["active_session_monitor"] = available_monitors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the 5090 and concurrency follow-up")
    parser.add_argument("--lexmount-full", type=Path, required=True)
    parser.add_argument("--local-full", type=Path, required=True)
    parser.add_argument("--local-smoke", type=Path, required=True)
    parser.add_argument("--local-rerun", type=Path, required=True)
    parser.add_argument("--paired-audit", type=Path, required=True)
    parser.add_argument("--capacity-c10", type=Path, required=True)
    parser.add_argument("--capacity-c64", type=Path, required=True)
    parser.add_argument("--sessions-c10", type=Path)
    parser.add_argument("--sessions-c64", type=Path)
    parser.add_argument("--probe-en-c64", type=Path, required=True)
    parser.add_argument("--probe-zh-c64", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    payload = {
        "schema_version": 1,
        "local_5090_failure_rerun": analyze_local_rerun(
            load(args.lexmount_full),
            load(args.local_full),
            load(args.local_smoke),
            load(args.local_rerun),
            load(args.paired_audit),
            bootstrap_samples=args.bootstrap_samples,
        ),
        "lexmount_concurrency": analyze_capacity(
            load(args.capacity_c10),
            load(args.capacity_c64),
            load(args.sessions_c10) if args.sessions_c10 else None,
            load(args.sessions_c64) if args.sessions_c64 else None,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "raw_session_capacity": {
            "profiles": {
                "en": compact_probe(load(args.probe_en_c64)),
                "zh": compact_probe(load(args.probe_zh_c64)),
            },
            "cleanup_caveat": (
                "the original probe's immediate sessions_after field was invalidated by "
                "sessions that activated after process exit; active sessions were then "
                "manually reconciled to zero and the probe cleanup logic was fixed"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
