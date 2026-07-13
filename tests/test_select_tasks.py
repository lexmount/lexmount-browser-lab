from __future__ import annotations

from scripts.select_tasks import _allocate_strata, select_tasks


def test_allocate_strata_preserves_total() -> None:
    allocation = _allocate_strata({("zh", "T1"): 118, ("en", "T1"): 76, ("en", "T2"): 16}, 80)

    assert sum(allocation.values()) == 80
    assert all(value > 0 for value in allocation.values())


def test_select_tasks_is_deterministic_and_stratified() -> None:
    tasks = [
        {"id": index, "website_region": region, "task_type": task_type}
        for index, (region, task_type) in enumerate(
            [("zh", "T1")] * 6 + [("en", "T1")] * 4 + [("zh", "T2")] * 2,
            start=1,
        )
    ]

    first = select_tasks(tasks, 6, "seed")
    second = select_tasks(tasks, 6, "seed")

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6
