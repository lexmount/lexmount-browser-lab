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


def test_noninferiority_does_not_treat_two_shared_failures_as_sufficient_evidence() -> None:
    summaries = {"per_task": {"1": {"success": False}, "2": {"success": False}}}

    result = compare_pair(summaries, summaries, bootstrap_samples=100, seed=1)

    assert result["positive_outcome_coverage"] == {
        "both_success": 0,
        "at_least_one_success": 0,
        "both_failed": 2,
    }
    assert result["noninferiority"]["local_only_count"] == 0
    assert result["noninferiority"]["local_only_one_sided_95_upper_bound"] > 0.05
    assert result["noninferiority"]["passed"] is False


def test_noninferiority_passes_after_sixty_non_local_only_pairs() -> None:
    summaries = {
        "per_task": {str(index): {"success": True} for index in range(1, 61)}
    }

    result = compare_pair(summaries, summaries, bootstrap_samples=100, seed=1)

    assert result["noninferiority"]["local_only_count"] == 0
    assert result["noninferiority"]["local_only_one_sided_95_upper_bound"] < 0.05
    assert result["noninferiority"]["passed"] is True
