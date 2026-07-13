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


def _compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
    profile_results = probe.get("profile_results")
    compact_profiles = None
    if isinstance(profile_results, dict):
        compact_profiles = {
            profile: {
                "requested": result.get("requested"),
                "created": result.get("created"),
                "failed": result.get("failed"),
                "late_sessions_cleaned": result.get("late_sessions_cleaned"),
                "remaining_new_session_ids": result.get("remaining_new_session_ids"),
                "cleanup_error_count": len(result.get("cleanup_errors") or []),
                "create_seconds": result.get("create_seconds"),
                "success": result.get("success"),
            }
            for profile, result in profile_results.items()
        }
    return {
        "profile": probe.get("profile"),
        "requested": probe.get("requested"),
        "requested_total": probe.get("requested_total"),
        "created": probe.get("created"),
        "failed": probe.get("failed"),
        "create_seconds": probe.get("create_seconds"),
        "active_sessions": probe.get("active_sessions"),
        "target_observed": probe.get("target_observed"),
        "profile_results": compact_profiles,
        "residual_active_sessions": probe.get("residual_active_sessions"),
        "residual_ok": probe.get("residual_ok"),
        "monitor_error_count": len(probe.get("monitor_errors") or []),
        "cleanup_error_count": len(probe.get("cleanup_errors") or []),
        "success": probe.get("success"),
    }


def _sustainable(
    summary: dict[str, Any],
    monitor: dict[str, Any] | None,
    *,
    expected_concurrency: int | None = None,
) -> bool:
    counts = summary.get("counts") or {}
    resources = summary.get("resource_summary") or {}
    complete = (
        counts.get("planned") is not None
        and counts.get("trajectory") == counts.get("planned")
        and counts.get("judged") == counts.get("planned")
    )
    no_session_create_failures = (
        int((summary.get("error_task_counts") or {}).get("session_create", 0)) == 0
    )
    monitor_ok = monitor is None or (
        monitor.get("residual_ok") is True and not (monitor.get("errors") or [])
    )
    observed_max = (
        _nested_number(monitor, "active_sessions", "total", "max")
        if monitor is not None
        else None
    )
    observed_target = expected_concurrency is None or (
        monitor is not None
        and observed_max is not None
        and observed_max >= expected_concurrency
    )
    return bool(
        complete
        and no_session_create_failures
        and resources.get("guard_triggered") is None
        and monitor_ok
        and observed_target
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


def _nested_number(value: dict[str, Any], *path: str) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def _resource_ratio(higher: dict[str, Any], lower: dict[str, Any]) -> dict[str, float | None]:
    higher_metrics = (higher.get("resource_summary") or {}).get("metrics") or {}
    lower_metrics = (lower.get("resource_summary") or {}).get("metrics") or {}
    paths = {
        "cpu_cores_mean": ("cpu_cores_mean",),
        "pss_gib_mean": ("pss_gib", "mean"),
        "pss_gib_p95": ("pss_gib", "p95"),
        "chrome_pss_gib_mean": ("chrome_pss_gib", "mean"),
        "chrome_pss_gib_p95": ("chrome_pss_gib", "p95"),
        "memory_current_gib_p95": ("memory_current_gib", "p95"),
        "memory_peak_kernel_gib": ("memory_peak_kernel_gib",),
    }
    output: dict[str, float | None] = {}
    for label, path in paths.items():
        numerator = _nested_number(higher_metrics, *path)
        denominator = _nested_number(lower_metrics, *path)
        output[label] = (
            round(numerator / denominator, 6)
            if numerator is not None and denominator not in (None, 0)
            else None
        )
    return output


def analyze_capacity_matrix(
    lexmount_runs: dict[int, dict[str, Any]],
    local_runs: dict[int, dict[str, Any]],
    *,
    lexmount_sessions: dict[int, dict[str, Any]] | None = None,
    probes: dict[str, dict[str, Any]] | None = None,
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
    backend_resource_comparison: dict[str, Any] = {}
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
        backend_resource_comparison[str(concurrency)] = {
            "throughput_ratio_local_over_lexmount": round(
                local_runs[concurrency]["throughput_task_per_hour"]
                / lexmount_runs[concurrency]["throughput_task_per_hour"],
                6,
            ),
            "resource_ratio_local_over_lexmount": _resource_ratio(
                local_runs[concurrency], lexmount_runs[concurrency]
            ),
        }
        sustainability[str(concurrency)] = {
            "lexmount": _sustainable(
                lexmount_runs[concurrency],
                monitor,
                expected_concurrency=concurrency,
            ),
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
            "resource_ratio": {
                "lexmount": _resource_ratio(lexmount_runs[higher], lexmount_runs[lower]),
                "local": _resource_ratio(local_runs[higher], local_runs[lower]),
            },
        }

    result = {
        "schema_version": 1,
        "task_count": len(reference_ids),
        "same_task_set": True,
        "concurrencies": concurrencies,
        "arms": arms,
        "paired_backend_quality": paired_backend_quality,
        "backend_resource_comparison": backend_resource_comparison,
        "within_backend_scaling": scaling,
        "sustainable": sustainability,
    }
    if probes:
        result["raw_session_probes"] = {
            label: _compact_probe(probe) for label, probe in sorted(probes.items())
        }
    return result


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


def _parse_named_paths(values: list[str], argument: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{argument}: expected LABEL=PATH")
        if label in output:
            raise ValueError(f"{argument}: duplicate label {label}")
        output[label] = Path(raw_path)
    return output


def _load_named_paths(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {label: json.loads(path.read_text(encoding="utf-8")) for label, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a paired browser capacity matrix")
    parser.add_argument("--lexmount", action="append", required=True, metavar="N=PATH")
    parser.add_argument("--local", action="append", required=True, metavar="N=PATH")
    parser.add_argument("--lexmount-session", action="append", default=[], metavar="N=PATH")
    parser.add_argument("--probe", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_capacity_matrix(
        _load_paths(_parse_paths(args.lexmount, "--lexmount")),
        _load_paths(_parse_paths(args.local, "--local")),
        lexmount_sessions=_load_paths(_parse_paths(args.lexmount_session, "--lexmount-session")),
        probes=_load_named_paths(_parse_named_paths(args.probe, "--probe")),
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
