#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lexmount import Lexmount
from lexmount.exceptions import LexmountError

from .probe_sessions import PROFILE_ENV, _redact_error, _required_env, _session_counts


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("en", "zh", "total"):
        values = [int(sample[key]) for sample in samples if sample.get(key) is not None]
        output[key] = {
            "mean": round(statistics.fmean(values), 3) if values else None,
            "p95": _percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return output


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_session_count(
    profile: str, client: Lexmount, credentials: dict[str, str]
) -> tuple[str, int | None, str | None]:
    try:
        counts = _session_counts(client)
        return profile, int(counts["active"]) if counts else None, None
    except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        message = _redact_error(str(exc), credentials)
        return profile, None, f"{type(exc).__name__}: {message}"


def _sample_active_sessions(
    clients: dict[str, Lexmount], credentials: dict[str, dict[str, str]]
) -> tuple[dict[str, int | None], list[dict[str, str]]]:
    def active(profile: str) -> tuple[str, int | None, str | None]:
        return _active_session_count(profile, clients[profile], credentials[profile])

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        results = list(executor.map(active, clients))
    values = {profile: value for profile, value, _ in results}
    errors = [
        {"profile": profile, "message": error} for profile, _, error in results if error is not None
    ]
    total_values = [value for value in values.values() if value is not None]
    values["total"] = sum(total_values) if len(total_values) == len(clients) else None
    return values, errors


def _residual_sessions(
    baseline: dict[str, int | None], final: dict[str, int | None]
) -> dict[str, int | None]:
    return {
        key: final[key] - baseline[key]
        if baseline.get(key) is not None and final.get(key) is not None
        else None
        for key in ("en", "zh", "total")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample active Lexmount sessions")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")

    load_dotenv(args.env_file, override=False)
    credentials = {profile: _required_env(profile) for profile in sorted(PROFILE_ENV)}
    clients = {profile: Lexmount(**creds) for profile, creds in credentials.items()}
    started_at = datetime.now(UTC)
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    baseline, baseline_errors = _sample_active_sessions(clients, credentials)
    errors.extend({**error, "phase": "baseline"} for error in baseline_errors)
    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text("ready\n", encoding="utf-8")

    while _process_exists(args.watch_pid):
        sample_started = time.monotonic()
        values, sample_errors = _sample_active_sessions(clients, credentials)
        errors.extend({**error, "phase": "sample"} for error in sample_errors)
        samples.append(
            {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "en": values.get("en"),
                "zh": values.get("zh"),
                "total": values.get("total"),
            }
        )
        sleep_for = args.interval_seconds - (time.monotonic() - sample_started)
        if sleep_for > 0:
            time.sleep(sleep_for)

    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    final, final_errors = _sample_active_sessions(clients, credentials)
    errors.extend({**error, "phase": "final"} for error in final_errors)
    residual = _residual_sessions(baseline, final)

    payload = {
        "schema_version": 1,
        "watch_pid": args.watch_pid,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "sample_count": len(samples),
        "baseline_active_sessions": baseline,
        "final_active_sessions": final,
        "residual_active_sessions": residual,
        "residual_ok": all(value is not None and value <= 0 for value in residual.values()),
        "active_sessions": summarize_samples(samples),
        "errors": errors,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
