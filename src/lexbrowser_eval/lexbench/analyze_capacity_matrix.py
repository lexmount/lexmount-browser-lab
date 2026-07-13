#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compare_pair import compare_pair


def _task_ids(summary: dict[str, Any], name: str) -> set[str]:
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict) or not per_task:
        raise ValueError(f"{name}: missing or empty per_task")
    return set(per_task)


def _compact_arm(summary: dict[str, Any]) -> dict[str, Any]:
    resource = summary.get("resource_summary") or {}
    return {
        "run_id": Path(str(summary.get("run_dir") or "unknown")).name,
        "counts": summary.get("counts"),
        "rates": summary.get("rates"),
        "e2e_seconds": summary.get("e2e_seconds"),
        "steps": summary.get("steps"),
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


def _compact_session_monitor(monitor: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_seconds": monitor.get("duration_seconds"),
        "sample_count": monitor.get("sample_count"),
        "baseline_active_sessions": monitor.get("baseline_active_sessions"),
        "final_active_sessions": monitor.get("final_active_sessions"),
        "residual_active_sessions": monitor.get("residual_active_sessions"),
        "residual_ok": monitor.get("residual_ok"),
        "active_sessions": monitor.get("active_sessions"),
        "monitor_error_count": len(monitor.get("errors") or []),
    }


def _sustainable(summary: dict[str, Any], monitor: dict[str, Any] | None) -> bool:
    counts = summary.get("counts") or {}
    resources = summary.get("resource_summary") or {}
    complete = (
        counts.get("planned") is not None
        and counts.get("trajectory") == counts.get("planned")
        and counts.get("judged") == counts.get("planned")
    )
    no_session_create_failures = int(
        (summary.get("error_task_counts") or {}).get("session_create", 0)
    ) == 0
    monitor_ok = monitor is None or (
        monitor.get("residual_ok") is True and not (monitor.get("errors") or [])
    )
    return bool(
        complete
        and no_session_create_failures
        and resources.get("guard_triggered") is None
        and monitor_ok
    )


def _named_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    first_name: str,
    second_name: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    comparison = compare_pair(
        first,
        second,
        bootstrap_samples=bootstrap_samples,
    )
    difference = comparison["success_rate_difference"]
    table = comparison["paired_table"]
    return {
        "paired_tasks": comparison["paired_tasks"],
        "success": {
            first_name: comparison["success"]["lexmount"],
            second_name: comparison["success"]["local"],
        },
        "paired_table": {
            "both_success": table["both_success"],
            f"{first_name}_only": table["lexmount_only"],
            f"{second_name}_only": table["local_only"],
            "both_failed": table["both_failed"],
        },
        "success_rate_difference": {
            f"{first_name}_minus_{second_name}": difference["lexmount_minus_local"],
            "bootstrap_95_ci": difference["bootstrap_95_ci"],
            "bootstrap_samples": difference["bootstrap_samples"],
        },
    }


def analyze_capacity_matrix(
    lexmount_runs: dict[int, dict[str, Any]],
    local_runs: dict[int, dict[str, Any]],
    *,
    lexmount_sessions: dict[int, dict[str, Any]] | None = None,
    bootstrap_samples: int = 100_000,
) -> dict[str, Any]:
    if not lexmount_runs or set(lexmount_runs) != set(local_runs):
        raise ValueError("Lexmount and Local must provide the same non-empty concurrency set")
    if lexmount_sessions is not None and not set(lexmount_sessions).issubset(lexmount_runs):
        raise ValueError("session monitors contain an unknown concurrency")

    concurrencies = sorted(lexmount_runs)
    reference_ids = _task_ids(lexmount_runs[concurrencies[0]], "Lexmount reference")
    for concurrency in concurrencies:
        for backend, summary in (
            ("Lexmount", lexmount_runs[concurrency]),
            ("Local", local_runs[concurrency]),
        ):
            if _task_ids(summary, f"{backend} c{concurrency}") != reference_ids:
                raise ValueError("all capacity arms must contain the same task ids")

    arms: dict[str, Any] = {}
    paired_backend_quality: dict[str, Any] = {}
    sustainability: dict[str, Any] = {}
    for concurrency in concurrencies:
        monitor = (lexmount_sessions or {}).get(concurrency)
        arms[str(concurrency)] = {
            "lexmount": _compact_arm(lexmount_runs[concurrency]),
            "local": _compact_arm(local_runs[concurrency]),
            **(
                {"lexmount_session_monitor": _compact_session_monitor(monitor)}
                if monitor is not None
                else {}
            ),
        }
        paired_backend_quality[str(concurrency)] = compare_pair(
            lexmount_runs[concurrency],
            local_runs[concurrency],
            bootstrap_samples=bootstrap_samples,
        )
        sustainability[str(concurrency)] = {
            "lexmount": _sustainable(lexmount_runs[concurrency], monitor),
            "local": _sustainable(local_runs[concurrency], None),
        }

    scaling: dict[str, Any] = {}
    for lower, higher in zip(concurrencies, concurrencies[1:], strict=False):
        label = f"c{lower}_to_c{higher}"
        scaling[label] = {
            "lexmount_quality": _named_pair(
                lexmount_runs[higher],
                lexmount_runs[lower],
                first_name=f"c{higher}",
                second_name=f"c{lower}",
                bootstrap_samples=bootstrap_samples,
            ),
            "local_quality": _named_pair(
                local_runs[higher],
                local_runs[lower],
                first_name=f"c{higher}",
                second_name=f"c{lower}",
                bootstrap_samples=bootstrap_samples,
            ),
            "throughput_ratio": {
                backend: round(
                    runs[higher]["throughput_task_per_hour"]
                    / runs[lower]["throughput_task_per_hour"],
                    6,
                )
                for backend, runs in (
                    ("lexmount", lexmount_runs),
                    ("local", local_runs),
                )
            },
        }

    return {
        "schema_version": 1,
        "task_count": len(reference_ids),
        "same_task_set": True,
        "concurrencies": concurrencies,
        "arms": arms,
        "paired_backend_quality": paired_backend_quality,
        "within_backend_scaling": scaling,
        "sustainable": sustainability,
    }


def _parse_paths(values: list[str], argument: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for value in values:
        raw_concurrency, separator, raw_path = value.partition("=")
        if not separator or not raw_concurrency.isdigit() or int(raw_concurrency) < 1:
            raise ValueError(f"{argument}: expected CONCURRENCY=PATH")
        concurrency = int(raw_concurrency)
        if concurrency in output:
            raise ValueError(f"{argument}: duplicate concurrency {concurrency}")
        output[concurrency] = Path(raw_path)
    return output


def _load_paths(paths: dict[int, Path]) -> dict[int, dict[str, Any]]:
    return {
        concurrency: json.loads(path.read_text(encoding="utf-8"))
        for concurrency, path in paths.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a paired browser capacity matrix")
    parser.add_argument("--lexmount", action="append", required=True, metavar="N=PATH")
    parser.add_argument("--local", action="append", required=True, metavar="N=PATH")
    parser.add_argument("--lexmount-session", action="append", default=[], metavar="N=PATH")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_capacity_matrix(
        _load_paths(_parse_paths(args.lexmount, "--lexmount")),
        _load_paths(_parse_paths(args.local, "--local")),
        lexmount_sessions=_load_paths(
            _parse_paths(args.lexmount_session, "--lexmount-session")
        ),
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
