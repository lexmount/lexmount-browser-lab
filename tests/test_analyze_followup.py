from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.followup import (
    analyze_capacity,
    analyze_local_rerun,
    compact_probe,
)


def _summary(successes: set[str], *, planned: int = 3) -> dict:
    return {
        "run_dir": "/tmp/run",
        "counts": {
            "planned": planned,
            "trajectory": planned,
            "judged": planned,
            "success": len(successes),
        },
        "rates": {},
        "steps": {},
        "e2e_seconds": {},
        "throughput_task_per_hour": 10.0,
        "error_task_counts": {"session_create": 0},
        "agent_usage": {},
        "judge_usage": {},
        "resource_summary": {
            "return_code": 0,
            "guard_triggered": None,
            "metrics": {},
        },
        "per_task": {
            task_id: {"success": task_id in successes} for task_id in ("1", "2", "3")[:planned]
        },
    }


def test_analyze_local_rerun_builds_local_favoring_sensitivity() -> None:
    lexmount = _summary({"1", "2"})
    local = _summary({"1"})
    smoke = _summary({"2"})
    rerun = _summary({"2"})
    rerun["per_task"].pop("1")
    paired_audit = {
        "discordant": [
            {
                "task_id": "2",
                "outcome": "lexmount_only",
                "local": {"failure_category": "E1"},
            }
        ]
    }

    result = analyze_local_rerun(
        lexmount,
        local,
        smoke,
        rerun,
        paired_audit,
        bootstrap_samples=100,
    )

    assert result["recovered_task_ids"] == ["2"]
    assert result["recovered_from_original_state"] == {
        "lexmount_only": ["2"],
        "both_failed": [],
    }
    comparison = result["local_favoring_sensitivity"]["comparison"]
    assert comparison["success"] == {"lexmount": 2, "local": 2}
    assert result["original_environment_loser_followup"] == {
        "tasks": 1,
        "recovered": 1,
        "still_failed": 0,
        "recovered_task_ids": ["2"],
        "source": "original Judge primary category E1/E2/E3",
    }


def test_analyze_local_rerun_rejects_unpaired_full_runs() -> None:
    lexmount = _summary({"1", "2"})
    lexmount["per_task"].pop("2")
    local = _summary({"1"})
    smoke = _summary(set())
    rerun = _summary({"2"})
    rerun["per_task"] = {"2": rerun["per_task"]["2"]}

    with pytest.raises(ValueError, match="missing from lexmount: \\['2'\\]"):
        analyze_local_rerun(
            lexmount,
            local,
            smoke,
            rerun,
            bootstrap_samples=100,
        )


def test_analyze_capacity_requires_and_compares_same_tasks() -> None:
    c10 = _summary({"1"})
    c64 = _summary({"1", "2"})
    sessions = {
        "sample_count": 2,
        "duration_seconds": 10.0,
        "active_sessions": {"total": {"mean": 2.0, "p95": 3, "max": 3}},
        "errors": [],
    }

    result = analyze_capacity(
        c10,
        c64,
        sessions,
        sessions,
        bootstrap_samples=100,
    )

    assert result["same_task_set"] is True
    assert result["paired_quality"]["success"] == {"c64": 2, "c10": 1}
    assert result["paired_quality"]["paired_table"] == {
        "both_success": 1,
        "c64_only": 1,
        "c10_only": 0,
        "both_failed": 1,
    }
    assert result["sustainable_before_external_session_check"] == {
        "c10": True,
        "c64": True,
    }
    assert result["active_session_monitor"]["c64"]["active_sessions"]["total"]["max"] == 3


def test_compact_probe_drops_failure_messages_and_immediate_cleanup_claim() -> None:
    result = compact_probe(
        {
            "profile": "en",
            "requested": 64,
            "created": 25,
            "failed": 39,
            "poll_timeout_seconds": 180,
            "create_seconds": {"mean": 73.7, "p95": 153.2, "max": 153.4},
            "sessions_before": {"active": 0},
            "sessions_after": {"active": 0},
            "failures": [{"message": "session id"}],
        }
    )

    assert result["created_within_timeout"] == 25
    assert "sessions_after" not in result
    assert "failures" not in result
