from __future__ import annotations

from lexbrowser_eval.lexbench.compare_pair import compare_pair


def test_compare_pair_builds_paired_table() -> None:
    lexmount = {
        "per_task": {
            "1": {"success": True},
            "2": {"success": True},
            "3": {"success": False},
        }
    }
    local = {
        "per_task": {
            "1": {"success": True},
            "2": {"success": False},
            "3": {"success": True},
        }
    }

    result = compare_pair(lexmount, local, bootstrap_samples=100, seed=1)

    assert result["paired_tasks"] == 3
    assert result["paired_table"] == {
        "both_success": 1,
        "lexmount_only": 1,
        "local_only": 1,
        "both_failed": 0,
    }
    assert result["success_rate_difference"]["lexmount_minus_local"] == 0.0
