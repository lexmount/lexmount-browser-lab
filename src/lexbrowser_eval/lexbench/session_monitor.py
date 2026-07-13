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

from .probe_sessions import PROFILE_ENV, _required_env, _session_counts


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample active Lexmount sessions")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    load_dotenv(args.env_file, override=False)
    clients = {profile: Lexmount(**_required_env(profile)) for profile in sorted(PROFILE_ENV)}
    started_at = datetime.now(UTC)
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    while _process_exists(args.watch_pid):
        sample_started = time.monotonic()

        def active(profile: str) -> tuple[str, int | None, str | None]:
            try:
                counts = _session_counts(clients[profile])
                return profile, int(counts["active"]) if counts else None, None
            except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
                return profile, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            results = list(executor.map(active, clients))
        values = {profile: value for profile, value, _ in results}
        for profile, _, error in results:
            if error:
                errors.append({"profile": profile, "message": error})
        total_values = [value for value in values.values() if value is not None]
        samples.append(
            {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "en": values.get("en"),
                "zh": values.get("zh"),
                "total": sum(total_values) if len(total_values) == len(clients) else None,
            }
        )
        sleep_for = args.interval_seconds - (time.monotonic() - sample_started)
        if sleep_for > 0:
            time.sleep(sleep_for)

    payload = {
        "schema_version": 1,
        "watch_pid": args.watch_pid,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "sample_count": len(samples),
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
