from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.compare_repeated_pairs import compare_repeated_pairs


def summary(values: list[int]) -> dict:
    return {
        "per_task": {
            str(index): {"success": bool(value)} for index, value in enumerate(values, start=1)
        }
    }


def test_compare_repeated_pairs_clusters_bootstrap_by_task() -> None:
    result = compare_repeated_pairs(
        [summary([1, 1, 0, 0]), summary([1, 0, 1, 0])],
        [summary([1, 0, 0, 0]), summary([0, 0, 1, 0])],
        labels=["local-first", "lexmount-first"],
        bootstrap_samples=100,
        seed=1,
    )

    assert result["paired_tasks"] == 4
    assert result["paired_attempts"] == 8
    assert result["success_attempts"] == {"lexmount": 4, "local": 2}
    assert result["clustered_difference"]["lexmount_minus_local"] == 0.25
    assert result["clustered_difference"]["cluster_unit"] == "task_id"
    assert result["replicate_difference_range"]["span"] == 0.0
    assert result["repeatability"]["lexmount"]["unanimous_rate"] == 0.5


def test_compare_repeated_pairs_rejects_different_task_coverage() -> None:
    with pytest.raises(ValueError, match="task coverage differs"):
        compare_repeated_pairs(
            [summary([1, 0]), summary([1])],
            [summary([0, 0]), summary([0, 0])],
            bootstrap_samples=10,
        )
