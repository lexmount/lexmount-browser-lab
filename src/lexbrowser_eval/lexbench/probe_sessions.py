#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lexmount import Lexmount
from lexmount.exceptions import LexmountError, SessionNotFoundError

PROFILE_ENV = {
    "en": ("LEXMOUNT_API_KEY", "LEXMOUNT_PROJECT_ID", "LEXMOUNT_BASE_URL"),
    "zh": ("LEXMOUNT_CN_API_KEY", "LEXMOUNT_CN_PROJECT_ID", "LEXMOUNT_CN_BASE_URL"),
}

SESSION_ID_RE = re.compile(r"\bsession_[A-Za-z0-9_-]+\b")
INACTIVE_SESSION_STATUSES = {"closed", "deleted", "failed", "stopped", "terminated"}


@dataclass
class SessionHandle:
    client: Lexmount
    session: Any
    session_id: str
    create_seconds: float


def _required_env(profile: str) -> dict[str, str]:
    names = PROFILE_ENV[profile]
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"missing environment variables: {', '.join(missing)}")
    base_url = values[names[2]]
    if "://" not in base_url:
        base_url = f"https://{base_url}"
    return {"api_key": values[names[0]], "project_id": values[names[1]], "base_url": base_url}


def _response_items(response: Any) -> list[Any] | None:
    if isinstance(response, (list, tuple)):
        return list(response)
    for field in ("items", "sessions", "data"):
        value = getattr(response, field, None)
        if isinstance(value, (list, tuple)):
            return list(value)
    try:
        return list(response)
    except TypeError:
        return None


def _session_items(client: Lexmount) -> list[Any] | None:
    return _response_items(client.sessions.list())


def _session_id(session: Any) -> str:
    return str(getattr(session, "session_id", None) or getattr(session, "id", None) or "")


def _session_status(session: Any) -> str:
    return (
        str(getattr(session, "status", None) or getattr(session, "state", None) or "unknown")
        .strip()
        .lower()
    )


def _active_session_ids(client: Lexmount) -> set[str]:
    return {
        _session_id(item)
        for item in (_session_items(client) or [])
        if _session_id(item) and _session_status(item) not in INACTIVE_SESSION_STATUSES
    }


def _session_counts(client: Lexmount) -> dict[str, Any] | None:
    response = client.sessions.list()
    pagination = getattr(response, "pagination", None)
    if pagination is not None:
        active = int(getattr(pagination, "active_count", 0))
        closed = int(getattr(pagination, "closed_count", 0))
        return {
            "active": active,
            "total_records": int(getattr(pagination, "total_count", active + closed)),
            "statuses": {"active": active, "closed": closed},
            "page_size": int(getattr(pagination, "page_size", len(response))),
            "total_pages": int(getattr(pagination, "total_pages", 1)),
        }
    items = _response_items(response)
    if items is None:
        return None
    statuses: dict[str, int] = {}
    active = 0
    for item in items:
        status = _session_status(item)
        statuses[status] = statuses.get(status, 0) + 1
        if status not in INACTIVE_SESSION_STATUSES:
            active += 1
    return {"active": active, "total_records": len(items), "statuses": statuses}


def _create_one(
    creds: dict[str, str], barrier: threading.Barrier, poll_timeout_seconds: float
) -> SessionHandle:
    client = Lexmount(**creds, timeout=min(60.0, poll_timeout_seconds))
    barrier.wait()
    started = time.monotonic()
    session = client.sessions.create(
        browser_mode="normal",
        official_proxy=False,
        poll_timeout_sec=poll_timeout_seconds,
    )
    elapsed = time.monotonic() - started
    session_id = _session_id(session)
    if not session_id:
        session.close()
        raise RuntimeError("session create returned no id")
    return SessionHandle(
        client=client, session=session, session_id=session_id, create_seconds=elapsed
    )


def _cleanup_one(handle: SessionHandle) -> str | None:
    errors: list[str] = []
    try:
        handle.session.close()
    except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        errors.append(f"close: {type(exc).__name__}: {exc}")
    try:
        handle.client.sessions.delete(session_id=handle.session_id)
    except SessionNotFoundError:
        pass
    except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        errors.append(f"delete: {type(exc).__name__}: {exc}")
    return "; ".join(errors) or None


def _cleanup_session_id(client: Lexmount, session_id: str) -> tuple[bool, str | None]:
    try:
        client.sessions.delete(session_id=session_id)
    except SessionNotFoundError:
        return False, None
    except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        return False, f"delete {session_id}: {type(exc).__name__}: {exc}"
    return True, None


def _session_ids_from_error(message: str) -> set[str]:
    return set(SESSION_ID_RE.findall(message))


def _reconcile_new_sessions(
    client: Lexmount,
    before_ids: set[str],
    known_session_ids: set[str],
    successful_session_ids: set[str],
    grace_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    if grace_seconds < 0:
        raise ValueError("cleanup grace seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("cleanup poll seconds must be positive")

    deadline = time.monotonic() + grace_seconds
    observed_new_ids: set[str] = set()
    cleanup_errors: set[str] = set()
    pending_known_ids = set(known_session_ids)
    polls = 0

    # Retry known IDs for the whole grace period. A timed-out create may not be
    # visible in sessions.list yet, but its ID is still present in the SDK error.
    while True:
        polls += 1
        current_ids = _active_session_ids(client) - before_ids
        observed_new_ids.update(current_ids)

        candidates = pending_known_ids | current_ids
        for session_id in sorted(candidates):
            deleted, error = _cleanup_session_id(client, session_id)
            if deleted:
                pending_known_ids.discard(session_id)
            if error:
                cleanup_errors.add(error)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_seconds, remaining))

    # One final list/delete/list pass closes the race at the grace boundary.
    final_new_ids = _active_session_ids(client) - before_ids
    observed_new_ids.update(final_new_ids)
    for session_id in sorted(pending_known_ids | final_new_ids):
        deleted, error = _cleanup_session_id(client, session_id)
        if deleted:
            pending_known_ids.discard(session_id)
        if error:
            cleanup_errors.add(error)
    remaining_new_ids = _active_session_ids(client) - before_ids

    return {
        "polls": polls,
        "cleanup_errors": sorted(cleanup_errors),
        "late_session_ids_cleaned": sorted(observed_new_ids - successful_session_ids),
        "remaining_new_session_ids": sorted(remaining_new_ids),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _redact_error(message: str, creds: dict[str, str]) -> str:
    redacted = message
    for value in creds.values():
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def run_probe(
    profile: str,
    count: int,
    hold_seconds: float,
    poll_timeout_seconds: float,
    cleanup_grace_seconds: float = 120.0,
    cleanup_poll_seconds: float = 5.0,
) -> dict[str, Any]:
    if count < 1 or count > 200:
        raise ValueError("count must be between 1 and 200")
    creds = _required_env(profile)
    control_client = Lexmount(**creds)
    before_ids = _active_session_ids(control_client)
    before = _session_counts(control_client)
    barrier = threading.Barrier(count)
    handles: list[SessionHandle] = []
    failures: list[dict[str, str]] = []
    failure_session_ids: set[str] = set()
    cleanup_errors: list[str] = []
    reconciliation: dict[str, Any] = {
        "polls": 0,
        "cleanup_errors": [],
        "late_session_ids_cleaned": [],
        "remaining_new_session_ids": [],
    }
    started = time.monotonic()

    try:
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [
                executor.submit(_create_one, creds, barrier, poll_timeout_seconds)
                for _ in range(count)
            ]
            for future in as_completed(futures):
                try:
                    handles.append(future.result())
                except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
                    message = _redact_error(str(exc), creds)
                    session_ids = _session_ids_from_error(message)
                    failure_session_ids.update(session_ids)
                    failures.append(
                        {
                            "type": type(exc).__name__,
                            "message": message,
                        }
                    )
        if handles and hold_seconds > 0:
            time.sleep(hold_seconds)
    finally:
        with ThreadPoolExecutor(max_workers=max(1, min(count, 32))) as executor:
            for error in executor.map(_cleanup_one, handles):
                if error:
                    cleanup_errors.append(_redact_error(error, creds))
        effective_grace_seconds = cleanup_grace_seconds if failures else 0.0
        reconciliation = _reconcile_new_sessions(
            control_client,
            before_ids,
            failure_session_ids,
            {handle.session_id for handle in handles},
            effective_grace_seconds,
            cleanup_poll_seconds,
        )
        cleanup_errors.extend(
            _redact_error(error, creds) for error in reconciliation["cleanup_errors"]
        )

    elapsed = time.monotonic() - started
    after = _session_counts(control_client)
    latencies = [handle.create_seconds for handle in handles]
    residual_ok = (
        before is None or after is None or int(after["active"]) <= int(before["active"])
    ) and not reconciliation["remaining_new_session_ids"]
    return {
        "profile": profile,
        "requested": count,
        "created": len(handles),
        "failed": len(failures),
        "hold_seconds": hold_seconds,
        "poll_timeout_seconds": poll_timeout_seconds,
        "cleanup_grace_seconds": cleanup_grace_seconds if failures else 0.0,
        "cleanup_polls": reconciliation["polls"],
        "elapsed_seconds": round(elapsed, 3),
        "create_seconds": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p95": round(_percentile(latencies, 0.95) or 0.0, 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "sessions_before": before,
        "sessions_after": after,
        "failure_session_ids": sorted(failure_session_ids),
        "late_sessions_cleaned": len(reconciliation["late_session_ids_cleaned"]),
        "late_session_ids_cleaned": reconciliation["late_session_ids_cleaned"],
        "remaining_new_session_ids": reconciliation["remaining_new_session_ids"],
        "cleanup_errors": cleanup_errors,
        "failures": failures,
        "success": len(handles) == count and not cleanup_errors and residual_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe simultaneous Lexmount session capacity")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_ENV), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=120.0)
    parser.add_argument("--cleanup-poll-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    result = run_probe(
        args.profile,
        args.count,
        args.hold_seconds,
        args.poll_timeout_seconds,
        args.cleanup_grace_seconds,
        args.cleanup_poll_seconds,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
