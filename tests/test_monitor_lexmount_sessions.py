from __future__ import annotations

import lexbrowser_eval.lexbench.session_monitor as session_monitor
from lexbrowser_eval.lexbench.session_monitor import _active_session_count, summarize_samples


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


def test_active_session_count_redacts_credentials(monkeypatch) -> None:
    secret = "secret-api-key"

    def raise_with_secret(_client) -> None:
        raise RuntimeError(f"request using {secret} failed")

    monkeypatch.setattr(session_monitor, "_session_counts", raise_with_secret)

    profile, count, error = _active_session_count("en", object(), {"api_key": secret})

    assert profile == "en"
    assert count is None
    assert error == "RuntimeError: request using <redacted> failed"
