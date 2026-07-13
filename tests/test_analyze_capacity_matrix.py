from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.analyze_capacity_matrix import analyze_capacity_matrix


def summary(
    successes: list[int],
    *,
    throughput: float,
    resource_scale: float = 1,
    guard: str | None = None,
) -> dict:
    return {
        "run_dir": "/tmp/run",
        "counts": {
            "planned": len(successes),
            "trajectory": len(successes),
            "judged": len(successes),
            "success": sum(successes),
        },
        "per_task": {
            str(index): {"success": bool(value)} for index, value in enumerate(successes, start=1)
        },
        "throughput_task_per_hour": throughput,
        "error_task_counts": {"session_create": 0},
        "resource_summary": {
            "guard_triggered": guard,
            "metrics": {
                "cpu_cores_mean": resource_scale,
                "pss_gib": {"mean": resource_scale * 2, "p95": resource_scale * 3},
                "chrome_pss_gib": {
                    "mean": resource_scale * 0.5,
                    "p95": resource_scale,
                },
                "memory_current_gib": {"p95": resource_scale * 4},
                "memory_peak_kernel_gib": resource_scale * 5,
            },
        },
    }


def test_analyze_capacity_matrix_compares_backends_and_scaling() -> None:
    lexmount = {
        16: summary([1, 0, 1, 0], throughput=100),
        32: summary([1, 1, 1, 0], throughput=160, resource_scale=2),
    }
    local = {
        16: summary([1, 1, 0, 0], throughput=80),
        32: summary([1, 0, 0, 0], throughput=120, resource_scale=1.5),
    }
    sessions = {
        16: {
            "residual_ok": True,
            "errors": [],
            "active_sessions": {"total": {"max": 16}},
        },
        32: {
            "residual_ok": True,
            "errors": [],
            "active_sessions": {"total": {"max": 32}},
        },
    }

    result = analyze_capacity_matrix(
        lexmount,
        local,
        lexmount_sessions=sessions,
        probes={
            "balanced64": {
                "requested": {"en": 32, "zh": 32},
                "requested_total": 64,
                "target_observed": True,
                "active_sessions": {"total": {"max": 64}},
                "residual_active_sessions": {"total": 0},
                "residual_ok": True,
                "monitor_errors": [],
                "profile_results": {
                    profile: {
                        "requested": 32,
                        "created": 32,
                        "failed": 0,
                        "cleanup_errors": [],
                        "success": True,
                    }
                    for profile in ("en", "zh")
                },
                "success": True,
            }
        },
        bootstrap_samples=100,
    )

    assert result["task_count"] == 4
    assert result["paired_backend_quality"]["32"]["success"] == {
        "lexmount": 3,
        "local": 1,
    }
    assert result["backend_resource_comparison"]["16"] == {
        "throughput_ratio_local_over_lexmount": 0.8,
        "resource_ratio_local_over_lexmount": {
            "cpu_cores_mean": 1.0,
            "pss_gib_mean": 1.0,
            "pss_gib_p95": 1.0,
            "chrome_pss_gib_mean": 1.0,
            "chrome_pss_gib_p95": 1.0,
            "memory_current_gib_p95": 1.0,
            "memory_peak_kernel_gib": 1.0,
        },
    }
    assert result["backend_resource_comparison"]["32"] == {
        "throughput_ratio_local_over_lexmount": 0.75,
        "resource_ratio_local_over_lexmount": {
            "cpu_cores_mean": 0.75,
            "pss_gib_mean": 0.75,
            "pss_gib_p95": 0.75,
            "chrome_pss_gib_mean": 0.75,
            "chrome_pss_gib_p95": 0.75,
            "memory_current_gib_p95": 0.75,
            "memory_peak_kernel_gib": 0.75,
        },
    }
    assert result["within_backend_scaling"]["c16_to_c32"]["throughput_ratio"] == {
        "lexmount": 1.6,
        "local": 1.5,
    }
    assert result["within_backend_scaling"]["c16_to_c32"]["resource_ratio"] == {
        "lexmount": {
            "cpu_cores_mean": 2.0,
            "pss_gib_mean": 2.0,
            "pss_gib_p95": 2.0,
            "chrome_pss_gib_mean": 2.0,
            "chrome_pss_gib_p95": 2.0,
            "memory_current_gib_p95": 2.0,
            "memory_peak_kernel_gib": 2.0,
        },
        "local": {
            "cpu_cores_mean": 1.5,
            "pss_gib_mean": 1.5,
            "pss_gib_p95": 1.5,
            "chrome_pss_gib_mean": 1.5,
            "chrome_pss_gib_p95": 1.5,
            "memory_current_gib_p95": 1.5,
            "memory_peak_kernel_gib": 1.5,
        },
    }
    assert result["sustainable"]["32"] == {"lexmount": True, "local": True}
    assert result["arms"]["32"]["lexmount_session_monitor"]["active_sessions"] == {
        "total": {"max": 32}
    }
    assert result["raw_session_probes"]["balanced64"]["active_sessions"] == {"total": {"max": 64}}
    assert result["raw_session_probes"]["balanced64"]["profile_results"]["en"] == {
        "requested": 32,
        "created": 32,
        "failed": 0,
        "late_sessions_cleaned": None,
        "remaining_new_session_ids": None,
        "cleanup_error_count": 0,
        "create_seconds": None,
        "success": True,
    }


def test_analyze_capacity_matrix_requires_same_tasks() -> None:
    lexmount = {16: summary([1, 0], throughput=100)}
    local = {16: summary([1, 0, 1], throughput=100)}

    with pytest.raises(ValueError, match="same task ids"):
        analyze_capacity_matrix(lexmount, local)


def test_analyze_capacity_matrix_marks_guarded_arm_unsustainable() -> None:
    lexmount = {16: summary([1, 0], throughput=100)}
    local = {16: summary([1, 0], throughput=100, guard="host memory")}

    result = analyze_capacity_matrix(lexmount, local, bootstrap_samples=10)

    assert result["sustainable"]["16"]["local"] is False


def test_analyze_capacity_matrix_requires_observed_lexmount_concurrency() -> None:
    lexmount = {16: summary([1, 0], throughput=100)}
    local = {16: summary([1, 0], throughput=100)}
    sessions = {
        16: {
            "residual_ok": True,
            "errors": [],
            "active_sessions": {"total": {"max": 15}},
        }
    }

    result = analyze_capacity_matrix(
        lexmount,
        local,
        lexmount_sessions=sessions,
        bootstrap_samples=10,
    )

    assert result["sustainable"]["16"]["lexmount"] is False


def test_analyze_capacity_matrix_requires_lexmount_session_monitor() -> None:
    lexmount = {16: summary([1, 0], throughput=100)}
    local = {16: summary([1, 0], throughput=100)}

    result = analyze_capacity_matrix(lexmount, local, bootstrap_samples=10)

    assert result["sustainable"]["16"]["lexmount"] is False


def test_analyze_capacity_matrix_handles_missing_and_zero_throughput() -> None:
    lexmount = {
        16: summary([1, 0], throughput=0),
        32: summary([1, 0], throughput=100),
    }
    local = {
        16: summary([1, 0], throughput=50),
        32: summary([1, 0], throughput=100),
    }
    del local[32]["throughput_task_per_hour"]

    result = analyze_capacity_matrix(lexmount, local, bootstrap_samples=10)

    assert (
        result["backend_resource_comparison"]["16"][
            "throughput_ratio_local_over_lexmount"
        ]
        is None
    )
    assert (
        result["backend_resource_comparison"]["32"][
            "throughput_ratio_local_over_lexmount"
        ]
        is None
    )
    assert result["within_backend_scaling"]["c16_to_c32"]["throughput_ratio"] == {
        "lexmount": None,
        "local": None,
    }
