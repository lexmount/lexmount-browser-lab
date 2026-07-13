from __future__ import annotations

from lexbrowser_eval.lexbench.session_monitor import summarize_samples


def test_summarize_samples_reports_actual_active_concurrency() -> None:
    result = summarize_samples(
        [
            {"en": 4, "zh": 6, "total": 10},
            {"en": 25, "zh": 36, "total": 61},
            {"en": 26, "zh": 38, "total": 64},
        ]
    )

    assert result["total"] == {"mean": 45.0, "p95": 64, "max": 64}
    assert result["en"]["max"] == 26
    assert result["zh"]["max"] == 38
