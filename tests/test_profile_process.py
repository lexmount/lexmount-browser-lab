from __future__ import annotations

from scripts.profile_process import accumulate_cpu_seconds, percentile, summarize_series


def test_process_profile_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.95) == 3.0


def test_process_profile_summary() -> None:
    assert summarize_series([1.0, 2.0, 3.0]) == {
        "mean": 2.0,
        "p95": 3.0,
        "max": 3.0,
    }


def test_process_profile_cpu_total_keeps_exited_children() -> None:
    maxima: dict[tuple[int, float], float] = {}

    first = accumulate_cpu_seconds(maxima, {(10, 1.0): 1.0, (11, 2.0): 2.0})
    second = accumulate_cpu_seconds(maxima, {(10, 1.0): 3.0})

    assert first == 3.0
    assert second == 5.0
