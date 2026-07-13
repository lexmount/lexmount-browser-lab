from __future__ import annotations

from types import SimpleNamespace

from lexbrowser_eval.lexbench.probe_sessions import (
    _reconcile_new_sessions,
    _session_counts,
    _session_ids_from_error,
)


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


def test_session_ids_from_timeout_error() -> None:
    message = (
        "Timed out waiting for session session_1783924677717_3grqqch03 "
        "to become active after 180.0s"
    )

    assert _session_ids_from_error(message) == {"session_1783924677717_3grqqch03"}


def test_reconcile_deletes_timeout_id_not_yet_listed() -> None:
    deleted: list[str] = []

    class FakeSessions:
        def list(self) -> list[object]:
            return []

        def delete(self, *, session_id: str) -> None:
            deleted.append(session_id)

    client = SimpleNamespace(sessions=FakeSessions())
    result = _reconcile_new_sessions(
        client,
        before_ids=set(),
        known_session_ids={"session_timeout_1"},
        successful_session_ids=set(),
        grace_seconds=0,
        poll_seconds=1,
    )

    assert deleted == ["session_timeout_1"]
    assert result == {
        "polls": 1,
        "cleanup_errors": [],
        "late_session_ids_cleaned": [],
        "remaining_new_session_ids": [],
    }


def test_reconcile_deletes_newly_visible_session() -> None:
    deleted: list[str] = []

    class FakeSessions:
        def __init__(self) -> None:
            self.list_calls = 0

        def list(self) -> list[object]:
            self.list_calls += 1
            if self.list_calls <= 2:
                return [SimpleNamespace(session_id="session_late_1")]
            return []

        def delete(self, *, session_id: str) -> None:
            deleted.append(session_id)

    client = SimpleNamespace(sessions=FakeSessions())
    result = _reconcile_new_sessions(
        client,
        before_ids=set(),
        known_session_ids=set(),
        successful_session_ids=set(),
        grace_seconds=0,
        poll_seconds=1,
    )

    assert deleted == ["session_late_1", "session_late_1"]
    assert result["late_session_ids_cleaned"] == ["session_late_1"]
    assert result["remaining_new_session_ids"] == []


def test_reconcile_ignores_new_closed_history_record() -> None:
    deleted: list[str] = []

    class FakeSessions:
        def list(self) -> list[object]:
            return [SimpleNamespace(session_id="session_closed_1", status="closed")]

        def delete(self, *, session_id: str) -> None:
            deleted.append(session_id)

    result = _reconcile_new_sessions(
        SimpleNamespace(sessions=FakeSessions()),
        before_ids=set(),
        known_session_ids=set(),
        successful_session_ids=set(),
        grace_seconds=0,
        poll_seconds=1,
    )

    assert deleted == []
    assert result["late_session_ids_cleaned"] == []
    assert result["remaining_new_session_ids"] == []
