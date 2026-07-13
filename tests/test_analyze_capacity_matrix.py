from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.analyze_capacity_matrix import analyze_capacity_matrix


def summary(successes: list[int], *, throughput: float, guard: str | None = None) -> dict:
    return {
        "run_dir": "/tmp/run",
        "counts": {
            "planned": len(successes),
            "trajectory": len(successes),
            "judged": len(successes),
            "success": sum(successes),
        },
        "per_task": {
            str(index): {"success": bool(value)}
            for index, value in enumerate(successes, start=1)
        },
        "throughput_task_per_hour": throughput,
        "error_task_counts": {"session_create": 0},
        "resource_summary": {"guard_triggered": guard, "metrics": {}},
    }


def test_analyze_capacity_matrix_compares_backends_and_scaling() -> None:
    lexmount = {
        16: summary([1, 0, 1, 0], throughput=100),
        32: summary([1, 1, 1, 0], throughput=160),
    }
    local = {
        16: summary([1, 1, 0, 0], throughput=80),
        32: summary([1, 0, 0, 0], throughput=120),
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
            "active_sessions": {"total": {"max": 31}},
        },
    }

    result = analyze_capacity_matrix(
        lexmount,
        local,
        lexmount_sessions=sessions,
        bootstrap_samples=100,
    )

    assert result["task_count"] == 4
    assert result["paired_backend_quality"]["32"]["success"] == {
        "lexmount": 3,
        "local": 1,
    }
    assert result["within_backend_scaling"]["c16_to_c32"]["throughput_ratio"] == {
        "lexmount": 1.6,
        "local": 1.5,
    }
    assert result["sustainable"]["32"] == {"lexmount": True, "local": True}
    assert result["arms"]["32"]["lexmount_session_monitor"]["active_sessions"] == {
        "total": {"max": 31}
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
