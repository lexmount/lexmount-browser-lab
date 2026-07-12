#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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


def _session_items(client: Lexmount) -> list[Any] | None:
    response = client.sessions.list()
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


def _session_id(session: Any) -> str:
    return str(getattr(session, "session_id", None) or getattr(session, "id", None) or "")


def _session_counts(client: Lexmount) -> dict[str, Any] | None:
    items = _session_items(client)
    if items is None:
        return None
    statuses: dict[str, int] = {}
    inactive = {"closed", "deleted", "failed", "stopped", "terminated"}
    active = 0
    for item in items:
        status = str(getattr(item, "status", None) or getattr(item, "state", None) or "unknown")
        status = status.strip().lower()
        statuses[status] = statuses.get(status, 0) + 1
        if status not in inactive:
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


def _cleanup_session_record(client: Lexmount, session: Any) -> str | None:
    session_id = _session_id(session)
    if not session_id:
        return "session record has no id"
    try:
        client.sessions.delete(session_id=session_id)
    except SessionNotFoundError:
        return None
    except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        return f"delete: {type(exc).__name__}: {exc}"
    return None


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
    profile: str, count: int, hold_seconds: float, poll_timeout_seconds: float
) -> dict[str, Any]:
    if count < 1 or count > 200:
        raise ValueError("count must be between 1 and 200")
    creds = _required_env(profile)
    control_client = Lexmount(**creds)
    before_items = _session_items(control_client) or []
    before_ids = {_session_id(item) for item in before_items if _session_id(item)}
    before = _session_counts(control_client)
    barrier = threading.Barrier(count)
    handles: list[SessionHandle] = []
    failures: list[dict[str, str]] = []
    cleanup_errors: list[str] = []
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
                    failures.append(
                        {
                            "type": type(exc).__name__,
                            "message": _redact_error(str(exc), creds),
                        }
                    )
        if handles and hold_seconds > 0:
            time.sleep(hold_seconds)
    finally:
        with ThreadPoolExecutor(max_workers=max(1, min(count, 32))) as executor:
            for error in executor.map(_cleanup_one, handles):
                if error:
                    cleanup_errors.append(_redact_error(error, creds))
        # Timed-out async creates already have server-side IDs that the SDK does
        # not return to this caller. Reconcile by list diff so saturation probes
        # cannot leave queued or active sessions behind.
        new_records = [
            item
            for item in (_session_items(control_client) or [])
            if _session_id(item) and _session_id(item) not in before_ids
        ]
        with ThreadPoolExecutor(max_workers=max(1, min(len(new_records), 32))) as executor:
            for error in executor.map(
                lambda item: _cleanup_session_record(control_client, item), new_records
            ):
                if error:
                    cleanup_errors.append(_redact_error(error, creds))

    elapsed = time.monotonic() - started
    after = _session_counts(control_client)
    latencies = [handle.create_seconds for handle in handles]
    residual_ok = before is None or after is None or int(after["active"]) <= int(before["active"])
    return {
        "profile": profile,
        "requested": count,
        "created": len(handles),
        "failed": len(failures),
        "hold_seconds": hold_seconds,
        "poll_timeout_seconds": poll_timeout_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "create_seconds": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p95": round(_percentile(latencies, 0.95) or 0.0, 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "sessions_before": before,
        "sessions_after": after,
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    result = run_probe(
        args.profile,
        args.count,
        args.hold_seconds,
        args.poll_timeout_seconds,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
