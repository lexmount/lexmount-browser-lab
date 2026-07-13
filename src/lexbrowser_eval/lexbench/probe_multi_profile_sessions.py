#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lexmount import Lexmount
from lexmount.exceptions import LexmountError

from .probe_sessions import (
    PROFILE_ENV,
    _redact_error,
    _required_env,
    _session_counts,
    run_probe,
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize_active_samples(
    samples: list[dict[str, Any]], profiles: list[str], requested_total: int
) -> dict[str, Any]:
    def summarize(key: str) -> dict[str, float | int | None]:
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        return {
            "mean": round(statistics.fmean(values), 3) if values else None,
            "p95": round(_percentile(values, 0.95) or 0.0, 3) if values else None,
            "max": int(max(values)) if values else None,
        }

    target_samples = [sample for sample in samples if sample.get("total") == requested_total]
    return {
        **{profile: summarize(profile) for profile in profiles},
        "total": summarize("total"),
        "target": requested_total,
        "target_sample_count": len(target_samples),
        "first_target_elapsed_seconds": (
            target_samples[0]["elapsed_seconds"] if target_samples else None
        ),
        "last_target_elapsed_seconds": (
            target_samples[-1]["elapsed_seconds"] if target_samples else None
        ),
    }


def _sample_profiles(
    clients: dict[str, Lexmount], credentials: dict[str, dict[str, str]]
) -> tuple[dict[str, int | None], list[dict[str, str]]]:
    values: dict[str, int | None] = {}
    errors: list[dict[str, str]] = []
    for profile, client in clients.items():
        try:
            counts = _session_counts(client)
            values[profile] = int(counts["active"]) if counts is not None else None
        except (LexmountError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            values[profile] = None
            errors.append(
                {
                    "profile": profile,
                    "message": f"{type(exc).__name__}: "
                    f"{_redact_error(str(exc), credentials[profile])}",
                }
            )
    known = [values[profile] for profile in clients if values[profile] is not None]
    values["total"] = sum(known) if len(known) == len(clients) else None
    return values, errors


def run_multi_profile_probe(
    counts: dict[str, int],
    *,
    hold_seconds: float,
    poll_timeout_seconds: float,
    sample_interval_seconds: float = 1.0,
    cleanup_grace_seconds: float = 120.0,
    cleanup_poll_seconds: float = 5.0,
) -> dict[str, Any]:
    if not counts or any(profile not in PROFILE_ENV for profile in counts):
        raise ValueError("counts must use known profiles")
    if any(count < 1 or count > 200 for count in counts.values()):
        raise ValueError("each profile count must be between 1 and 200")
    if sample_interval_seconds <= 0:
        raise ValueError("sample interval must be positive")

    profiles = sorted(counts)
    credentials = {profile: _required_env(profile) for profile in profiles}
    clients = {profile: Lexmount(**credentials[profile]) for profile in profiles}
    baseline, baseline_errors = _sample_profiles(clients, credentials)
    samples: list[dict[str, Any]] = []
    monitor_errors = [{**error, "phase": "baseline"} for error in baseline_errors]
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=len(profiles)) as executor:
        futures = {
            profile: executor.submit(
                run_probe,
                profile,
                counts[profile],
                hold_seconds,
                poll_timeout_seconds,
                cleanup_grace_seconds,
                cleanup_poll_seconds,
            )
            for profile in profiles
        }
        while not all(future.done() for future in futures.values()):
            sample_started = time.monotonic()
            values, errors = _sample_profiles(clients, credentials)
            samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    **values,
                }
            )
            monitor_errors.extend({**error, "phase": "sample"} for error in errors)
            sleep_for = sample_interval_seconds - (time.monotonic() - sample_started)
            if sleep_for > 0:
                time.sleep(sleep_for)
        profile_results = {profile: future.result() for profile, future in futures.items()}

    final, final_errors = _sample_profiles(clients, credentials)
    monitor_errors.extend({**error, "phase": "final"} for error in final_errors)
    residual = {
        key: final[key] - baseline[key]
        if baseline.get(key) is not None and final.get(key) is not None
        else None
        for key in [*profiles, "total"]
    }
    requested_total = sum(counts.values())
    active = summarize_active_samples(samples, profiles, requested_total)
    residual_ok = all(value is not None and value <= 0 for value in residual.values())
    target_observed = active["total"]["max"] == requested_total
    success = (
        all(result["success"] for result in profile_results.values())
        and target_observed
        and residual_ok
        and not monitor_errors
    )
    return {
        "schema_version": 1,
        "profiles": profiles,
        "requested": counts,
        "requested_total": requested_total,
        "hold_seconds": hold_seconds,
        "poll_timeout_seconds": poll_timeout_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "duration_seconds": round(time.monotonic() - started, 3),
        "baseline_active_sessions": baseline,
        "final_active_sessions": final,
        "residual_active_sessions": residual,
        "residual_ok": residual_ok,
        "active_sessions": active,
        "target_observed": target_observed,
        "monitor_errors": monitor_errors,
        "profile_results": profile_results,
        "success": success,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe simultaneous sessions across profiles")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--en-count", type=int, required=True)
    parser.add_argument("--zh-count", type=int, required=True)
    parser.add_argument("--hold-seconds", type=float, default=60.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=120.0)
    parser.add_argument("--cleanup-poll-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    result = run_multi_profile_probe(
        {"en": args.en_count, "zh": args.zh_count},
        hold_seconds=args.hold_seconds,
        poll_timeout_seconds=args.poll_timeout_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
        cleanup_poll_seconds=args.cleanup_poll_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "success": result["success"],
                "requested_total": result["requested_total"],
                "active_sessions": result["active_sessions"],
                "residual_active_sessions": result["residual_active_sessions"],
                "monitor_error_count": len(result["monitor_errors"]),
            }
        )
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
