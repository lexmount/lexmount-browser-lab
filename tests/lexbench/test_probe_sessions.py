from __future__ import annotations

from types import SimpleNamespace

from lexbrowser_eval.lexbench.probe_sessions import _session_counts


def test_session_counts_uses_pagination_totals() -> None:
    class FakeResponse:
        pagination = SimpleNamespace(
            active_count=7,
            closed_count=143,
            total_count=150,
            page_size=100,
            total_pages=2,
        )

        def __len__(self) -> int:
            return 100

    response = FakeResponse()
    client = SimpleNamespace(sessions=SimpleNamespace(list=lambda: response))

    assert _session_counts(client) == {
        "active": 7,
        "total_records": 150,
        "statuses": {"active": 7, "closed": 143},
        "page_size": 100,
        "total_pages": 2,
    }
