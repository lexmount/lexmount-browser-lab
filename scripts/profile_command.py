#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

GIB = 1024**3
CHROME_MARKERS = ("chrome", "chromium", "headless_shell")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def summarize_series(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def _read_cpu_usage_usec(cgroup: Path) -> int | None:
    try:
        for line in (cgroup / "cpu.stat").read_text(encoding="utf-8").splitlines():
            key, value = line.split(maxsplit=1)
            if key == "usage_usec":
                return int(value)
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None
    return None


def _read_host_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None
    return None


def _cgroup_pids(cgroup: Path) -> set[int]:
    pids: set[int] = set()
    for path in cgroup.rglob("cgroup.procs"):
        try:
            pids.update(int(value) for value in path.read_text().split() if value.isdigit())
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            continue
    return pids


def _process_pss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        return None
    return None


def _process_cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return ""


def _process_memory(cgroup: Path) -> tuple[int, int, int]:
    total_pss = 0
    chrome_pss = 0
    sampled = 0
    for pid in _cgroup_pids(cgroup):
        pss = _process_pss_bytes(pid)
        if pss is None:
            continue
        sampled += 1
        total_pss += pss
        cmdline = _process_cmdline(pid).lower()
        if any(marker in cmdline for marker in CHROME_MARKERS):
            chrome_pss += pss
    return total_pss, chrome_pss, sampled


def _gpu_sample() -> tuple[float | None, float | None, float | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None, None
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    rows: list[tuple[float, float, float]] = []
    for line in result.stdout.splitlines():
        try:
            utilization, memory_mib, power_w = [float(value.strip()) for value in line.split(",")]
        except (ValueError, TypeError):
            continue
        rows.append((utilization, memory_mib, power_w))
    if not rows:
        return None, None, None
    return tuple(statistics.fmean(row[index] for row in rows) for index in range(3))


_VLLM_METRIC = re.compile(
    r"^vllm:(num_requests_running|num_requests_waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


def _vllm_sample(url: str | None) -> tuple[float | None, float | None]:
    if not url:
        return None, None
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, TimeoutError, ValueError):
        return None, None
    values = {name: float(value) for name, value in _VLLM_METRIC.findall(text)}
    return values.get("num_requests_running"), values.get("num_requests_waiting")


def _control_group(unit: str, timeout_seconds: float = 20.0) -> Path:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ControlGroup", "--value"],
            capture_output=True,
            text=True,
            check=False,
        )
        value = result.stdout.strip()
        if value:
            path = Path("/sys/fs/cgroup") / value.lstrip("/")
            if path.exists():
                return path
        time.sleep(0.2)
    raise RuntimeError(f"systemd control group for {unit} was not created")


def _kill_unit(unit: str, sig: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "kill", "--kill-whom=all", f"--signal={sig}", unit],
        capture_output=True,
        check=False,
    )


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command in a measured systemd user scope")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--memory-max", default="46G")
    parser.add_argument("--min-host-available-gib", type=float, default=32.0)
    parser.add_argument("--planned-tasks", type=int)
    parser.add_argument("--vllm-metrics-url")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        raise SystemExit("[FAILED] cgroup v2 is required")
    for executable in ("systemd-run", "systemctl"):
        if shutil.which(executable) is None:
            raise SystemExit(f"[FAILED] {executable} is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.csv"
    summary_path = args.output_dir / "resource_summary.json"
    log_path = args.output_dir / "command.log"
    unit_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.label).strip("-")[:40] or "run"
    unit = f"lexbrowserenv-{unit_stem}-{uuid.uuid4().hex[:8]}.scope"
    systemd_command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={unit.removesuffix('.scope')}",
        f"--property=MemoryMax={args.memory_max}",
        *command,
    ]

    started_wall = datetime.now(UTC)
    started = time.monotonic()
    guard_triggered: str | None = None
    rows: list[dict[str, float | int | None]] = []
    previous_cpu: int | None = None
    previous_elapsed: float | None = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            systemd_command,
            cwd=args.cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            cgroup = _control_group(unit)
            while process.poll() is None:
                sample_started = time.monotonic()
                elapsed = sample_started - started
                cpu_usage = _read_cpu_usage_usec(cgroup)
                cpu_cores = None
                if (
                    cpu_usage is not None
                    and previous_cpu is not None
                    and previous_elapsed is not None
                ):
                    delta_seconds = elapsed - previous_elapsed
                    if delta_seconds > 0:
                        cpu_cores = (cpu_usage - previous_cpu) / 1_000_000 / delta_seconds
                total_pss, chrome_pss, process_count = _process_memory(cgroup)
                memory_current = _read_int(cgroup / "memory.current")
                memory_peak = _read_int(cgroup / "memory.peak")
                host_available = _read_host_available_bytes()
                gpu_util, gpu_memory_mib, gpu_power_w = _gpu_sample()
                vllm_running, vllm_waiting = _vllm_sample(args.vllm_metrics_url)
                rows.append(
                    {
                        "elapsed_seconds": elapsed,
                        "cpu_usage_usec": cpu_usage,
                        "cpu_cores": cpu_cores,
                        "process_count": process_count,
                        "pss_gib": total_pss / GIB,
                        "chrome_pss_gib": chrome_pss / GIB,
                        "memory_current_gib": memory_current / GIB
                        if memory_current is not None
                        else None,
                        "memory_peak_gib": memory_peak / GIB if memory_peak is not None else None,
                        "host_available_gib": host_available / GIB
                        if host_available is not None
                        else None,
                        "gpu_utilization_percent": gpu_util,
                        "gpu_memory_mib": gpu_memory_mib,
                        "gpu_power_w": gpu_power_w,
                        "vllm_running": vllm_running,
                        "vllm_waiting": vllm_waiting,
                    }
                )
                previous_cpu = cpu_usage
                previous_elapsed = elapsed

                if (
                    host_available is not None
                    and host_available < args.min_host_available_gib * GIB
                ):
                    guard_triggered = (
                        f"host MemAvailable fell below {args.min_host_available_gib:.1f} GiB"
                    )
                    _kill_unit(unit, "TERM")
                    time.sleep(5)
                    if process.poll() is None:
                        _kill_unit(unit, "KILL")
                    break
                sleep_for = args.interval - (time.monotonic() - sample_started)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            return_code = process.wait()
        except KeyboardInterrupt:
            guard_triggered = "operator interrupt"
            _kill_unit(unit, "TERM")
            time.sleep(2)
            _kill_unit(unit, "KILL")
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

    cpu_usage_values = series("cpu_usage_usec")
    average_cpu_cores = None
    cpu_elapsed_values = [
        float(row["elapsed_seconds"]) for row in rows if row["cpu_usage_usec"] is not None
    ]
    if len(cpu_usage_values) >= 2 and len(cpu_elapsed_values) >= 2:
        sampled_duration = cpu_elapsed_values[-1] - cpu_elapsed_values[0]
        if sampled_duration > 0:
            average_cpu_cores = (
                (cpu_usage_values[-1] - cpu_usage_values[0]) / 1_000_000 / sampled_duration
            )
    gpu_utilization = series("gpu_utilization_percent")
    vllm_running = series("vllm_running")
    vllm_waiting = series("vllm_waiting")

    summary = {
        "schema_version": 1,
        "label": args.label,
        "unit": unit,
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
        "memory_max": args.memory_max,
        "min_host_available_gib_guard": args.min_host_available_gib,
        "metrics": {
            "cpu_cores_mean": _round(average_cpu_cores),
            "pss_gib": {
                key: _round(value) for key, value in summarize_series(series("pss_gib")).items()
            },
            "chrome_pss_gib": {
                key: _round(value)
                for key, value in summarize_series(series("chrome_pss_gib")).items()
            },
            "memory_current_gib": {
                key: _round(value)
                for key, value in summarize_series(series("memory_current_gib")).items()
            },
            "memory_peak_kernel_gib": _round(max(series("memory_peak_gib"), default=None)),
            "host_available_gib_min": _round(min(series("host_available_gib"), default=None)),
            "gpu_utilization_percent_mean": _round(
                statistics.fmean(gpu_utilization) if gpu_utilization else None
            ),
            "gpu_idle_percent_mean": _round(
                100 - statistics.fmean(gpu_utilization) if gpu_utilization else None
            ),
            "gpu_memory_mib_mean": _round(
                statistics.fmean(series("gpu_memory_mib")) if series("gpu_memory_mib") else None
            ),
            "gpu_power_w_mean": _round(
                statistics.fmean(series("gpu_power_w")) if series("gpu_power_w") else None
            ),
            "vllm_running_nonzero_fraction": _round(
                sum(value > 0 for value in vllm_running) / len(vllm_running)
                if vllm_running
                else None
            ),
            "vllm_waiting_nonzero_fraction": _round(
                sum(value > 0 for value in vllm_waiting) / len(vllm_waiting)
                if vllm_waiting
                else None
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
