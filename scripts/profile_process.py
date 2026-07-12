#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

GIB = 1024**3
CHROME_MARKERS = ("chrome", "chromium", "headless_shell")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize_series(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _processes(root: psutil.Process) -> list[psutil.Process]:
    try:
        candidates = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        candidates = [root]
    return list({process.pid: process for process in candidates}.values())


ProcessIdentity = tuple[int, float]


def _process_tree_sample(
    root: psutil.Process,
) -> tuple[dict[ProcessIdentity, float], int, int, int]:
    cpu_seconds_by_process: dict[ProcessIdentity, float] = {}
    rss_bytes = 0
    chrome_rss_bytes = 0
    sampled = 0
    for process in _processes(root):
        try:
            times = process.cpu_times()
            identity = (process.pid, process.create_time())
            memory = process.memory_info()
            name = process.name().lower()
            cmdline = " ".join(process.cmdline()).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        sampled += 1
        cpu_seconds_by_process[identity] = float(times.user + times.system)
        rss_bytes += int(memory.rss)
        if any(marker in name or marker in cmdline for marker in CHROME_MARKERS):
            chrome_rss_bytes += int(memory.rss)
    return cpu_seconds_by_process, rss_bytes, chrome_rss_bytes, sampled


def accumulate_cpu_seconds(
    maxima: dict[ProcessIdentity, float], sample: dict[ProcessIdentity, float]
) -> float:
    for identity, cpu_seconds in sample.items():
        maxima[identity] = max(maxima.get(identity, 0.0), cpu_seconds)
    return sum(maxima.values())


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and profile a process tree on macOS/Linux")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--min-host-available-gib", type=float, default=6.0)
    parser.add_argument("--planned-tasks", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.csv"
    summary_path = args.output_dir / "resource_summary.json"
    log_path = args.output_dir / "command.log"
    started_wall = datetime.now(UTC)
    started = time.monotonic()
    guard_triggered: str | None = None
    rows: list[dict[str, float | int | None]] = []
    cpu_maxima: dict[ProcessIdentity, float] = {}

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        root = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                sample_started = time.monotonic()
                elapsed = sample_started - started
                cpu_sample, rss_bytes, chrome_rss_bytes, process_count = _process_tree_sample(root)
                cpu_seconds = accumulate_cpu_seconds(cpu_maxima, cpu_sample)
                host_available = int(psutil.virtual_memory().available)
                rows.append(
                    {
                        "elapsed_seconds": elapsed,
                        "cpu_usage_seconds": cpu_seconds,
                        "process_count": process_count,
                        "rss_gib": rss_bytes / GIB,
                        "chrome_rss_gib": chrome_rss_bytes / GIB,
                        "host_available_gib": host_available / GIB,
                    }
                )
                if host_available < args.min_host_available_gib * GIB:
                    guard_triggered = (
                        f"host MemAvailable fell below {args.min_host_available_gib:.1f} GiB"
                    )
                    _terminate_process_group(process)
                    break
                sleep_for = args.interval - (time.monotonic() - sample_started)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            return_code = process.wait()
        except KeyboardInterrupt:
            guard_triggered = "operator interrupt"
            _terminate_process_group(process)
            return_code = process.wait()

    ended_wall = datetime.now(UTC)
    duration = time.monotonic() - started
    if rows:
        with samples_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def series(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row[key] is not None]

    cpu_values = series("cpu_usage_seconds")
    elapsed_values = series("elapsed_seconds")
    average_cpu_cores = None
    if len(cpu_values) >= 2 and len(elapsed_values) >= 2:
        sampled_duration = elapsed_values[-1] - elapsed_values[0]
        if sampled_duration > 0:
            average_cpu_cores = (cpu_values[-1] - cpu_values[0]) / sampled_duration

    summary: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "platform": platform.platform(),
        "metric_scope": "process_tree",
        "memory_metric": "RSS",
        "started_at": started_wall.isoformat(),
        "ended_at": ended_wall.isoformat(),
        "duration_seconds": _round(duration, 3),
        "cwd": str(args.cwd.resolve()),
        "command": command,
        "return_code": return_code,
        "guard_triggered": guard_triggered,
        "sample_interval_seconds": args.interval,
        "sample_count": len(rows),
        "planned_tasks": args.planned_tasks,
        "planned_throughput_task_per_hour": _round(
            args.planned_tasks / duration * 3600 if args.planned_tasks and duration > 0 else None
        ),
        "min_host_available_gib_guard": args.min_host_available_gib,
        "metrics": {
            "cpu_cores_mean": _round(average_cpu_cores),
            "rss_gib": {
                key: _round(value) for key, value in summarize_series(series("rss_gib")).items()
            },
            "chrome_rss_gib": {
                key: _round(value)
                for key, value in summarize_series(series("chrome_rss_gib")).items()
            },
            "host_available_gib_min": _round(min(series("host_available_gib"), default=None)),
            "pss_gib": {"mean": None, "p95": None, "max": None},
            "chrome_pss_gib": {"mean": None, "p95": None, "max": None},
            "memory_peak_kernel_gib": None,
            "gpu_utilization_percent_mean": None,
            "gpu_idle_percent_mean": None,
            "gpu_memory_mib_mean": None,
            "gpu_power_w_mean": None,
            "vllm_running_nonzero_fraction": None,
            "vllm_waiting_nonzero_fraction": None,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
