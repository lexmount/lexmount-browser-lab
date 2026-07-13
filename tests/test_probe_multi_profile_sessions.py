from __future__ import annotations

from lexbrowser_eval.lexbench.probe_multi_profile_sessions import summarize_active_samples


def test_summarize_active_samples_requires_observed_target() -> None:
    samples = [
        {"elapsed_seconds": 0.0, "en": 0, "zh": 0, "total": 0},
        {"elapsed_seconds": 1.0, "en": 31, "zh": 32, "total": 63},
        {"elapsed_seconds": 2.0, "en": 32, "zh": 32, "total": 64},
        {"elapsed_seconds": 3.0, "en": 32, "zh": 32, "total": 64},
    ]

    result = summarize_active_samples(samples, ["en", "zh"], 64)

    assert result["en"]["max"] == 32
    assert result["zh"]["max"] == 32
    assert result["total"]["max"] == 64
    assert result["target_sample_count"] == 2
    assert result["first_target_elapsed_seconds"] == 2.0
    assert result["last_target_elapsed_seconds"] == 3.0


def test_summarize_active_samples_preserves_unknown_samples() -> None:
    samples = [
        {"elapsed_seconds": 0.0, "en": None, "zh": 2, "total": None},
        {"elapsed_seconds": 1.0, "en": 3, "zh": 2, "total": 5},
    ]

    result = summarize_active_samples(samples, ["en", "zh"], 6)

    assert result["en"] == {"mean": 3.0, "p95": 3.0, "max": 3}
    assert result["total"] == {"mean": 5.0, "p95": 5.0, "max": 5}
    assert result["target_sample_count"] == 0


def test_summarize_active_samples_accepts_counts_above_target() -> None:
    samples = [
        {"elapsed_seconds": 0.0, "en": 2, "zh": 1, "total": 3},
        {"elapsed_seconds": 1.0, "en": 5, "zh": 5, "total": 10},
        {"elapsed_seconds": 2.0, "en": 6, "zh": 5, "total": 11},
    ]

    result = summarize_active_samples(samples, ["en", "zh"], 10)

    assert result["target"] == 10
    assert result["target_sample_count"] == 2
    assert result["first_target_elapsed_seconds"] == 1.0
    assert result["last_target_elapsed_seconds"] == 2.0
