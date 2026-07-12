from __future__ import annotations

from scripts.profile_command import percentile, summarize_series


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.95) is None


def test_summarize_series() -> None:
    result = summarize_series([1.0, 2.0, 3.0])

    assert result == {"mean": 2.0, "p95": 3.0, "max": 3.0}
