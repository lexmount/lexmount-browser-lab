#!/usr/bin/env python3
"""Run the approved process-attributed LexBench stress matrix on the 5090 host.

The upstream checkout is treated as read-only. Repeated task IDs are isolated
in separate official timestamp directories, while one transient systemd
service/cgroup contains every replica in a cell. The external campaign process
samples that cgroup, so monitor overhead is not charged to the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .stress import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_TASK_IDS,
    EXPECTED_TASK_IDS_SHA256,
    RANDOM_SEED,
    TARGETS,
    build_official_replica_command,
    build_stress_manifest,
    canonical_manifest_sha256,
    validate_config_snapshot,
    validate_frozen_sample,
)
from .stress_monitor import (
    RESOURCE_CSV_FIELDS,
    ProcessTreeStats,
    calculate_cpu_cores,
    calculate_throughput_per_hour,
    read_cgroup_processes,
    read_cgroup_stats,
    read_host_memory,
    summarize_processes,
)

MODULE_NAME = "lexbrowser_eval.lexbench.qwen3_8b.campaign"


GIB = 1024**3
BASELINE_SECONDS = 60
COOLDOWN_SECONDS = 30
PSS_INTERVAL_SECONDS = 5
HOST_MEMORY_RESERVE_BYTES = 32 * GIB
MEMORY_BUDGET_HEADROOM_BYTES = 4 * GIB
MAX_CELL_MEMORY_BYTES = 46 * GIB
MIN_CELL_MEMORY_BYTES = 16 * GIB
MEMORY_HIGH_FRACTION = 0.85
MEMORY_STOP_FRACTION = 0.90
TASKS_MAX = 32768
OFFICIAL_COMMIT = "ccd5fcbdfb975257b2ce38161dc9bc2ab294b420"
GPU_CSV_FIELDS = (
    "timestamp_utc",
    "monotonic_seconds",
    "phase",
    "gpu_index",
    "gpu_uuid",
    "gpu_sm_percent",
    "gpu_idle_flag",
    "gpu_memory_used_mib",
    "gpu_power_w",
    "qwen_worker_present",
    "non_qwen_compute_pid_count",
    "compute_pid_fingerprint",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(path: pathlib.Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"timestamp_utc": utc_now(), "event": event, **fields},
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        )
        handle.flush()


def read_runtime_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        parsed = shlex.split(raw_value, comments=False, posix=True)
        values[name] = parsed[0] if parsed else ""
    return values


def qwen_metrics_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return f"{normalized}/metrics"


def read_qwen_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    suffixes = {
        "num_requests_running": "qwen_requests_running",
        "num_requests_waiting": "qwen_requests_waiting",
        "prompt_tokens_total": "qwen_prompt_tokens_total",
        "generation_tokens_total": "qwen_generation_tokens_total",
    }
    result = {target: 0.0 for target in suffixes.values()}
    found: set[str] = set()
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 2:
            continue
        metric = parts[0].split("{", 1)[0]
        for suffix, target in suffixes.items():
            if metric.endswith(suffix):
                result[target] += float(parts[-1])
                found.add(target)
    required = {"qwen_requests_running", "qwen_requests_waiting"}
    if not required.issubset(found):
        raise RuntimeError("Qwen metrics are missing running/waiting gauges")
    return result


def _qwen_pids() -> set[int]:
    result: set[int] = set()
    parents: dict[int, int] = {}
    for path in pathlib.Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (
                (path / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .lower()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        try:
            parents[int(path.name)] = int(
                next(
                    line.split(":", 1)[1]
                    for line in (path / "status").read_text().splitlines()
                    if line.startswith("PPid:")
                )
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration, ValueError):
            pass
        if "vllm" in command and ("qwen3-8b" in command or "qwen3_8b" in command):
            result.add(int(path.name))
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def gpu_rows(phase: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    qwen_pids = _qwen_pids()
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    non_qwen: list[str] = []
    qwen_present = False
    for raw in apps.splitlines():
        parts = [part.strip() for part in raw.split(",", 2)]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        if pid in qwen_pids:
            qwen_present = True
        else:
            non_qwen.append("|".join(parts))
    fingerprint = hashlib.sha256("\n".join(sorted(non_qwen)).encode()).hexdigest()
    now = time.monotonic()
    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 5:
            raise RuntimeError("Malformed nvidia-smi GPU row")
        sm = float(parts[2])
        rows.append(
            {
                "timestamp_utc": utc_now(),
                "monotonic_seconds": f"{now:.6f}",
                "phase": phase,
                "gpu_index": int(parts[0]),
                "gpu_uuid": parts[1],
                "gpu_sm_percent": sm,
                "gpu_idle_flag": int(sm == 0.0),
                "gpu_memory_used_mib": float(parts[3]),
                "gpu_power_w": float(parts[4]),
                "qwen_worker_present": int(qwen_present),
                "non_qwen_compute_pid_count": len(non_qwen),
                "compute_pid_fingerprint": fingerprint,
            }
        )
    return rows


def systemctl_properties(unit: str) -> dict[str, str]:
    output = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--no-page",
            "--property=ActiveState",
            "--property=SubState",
            "--property=Result",
            "--property=ExecMainStatus",
            "--property=ControlGroup",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    result: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def cheap_active_processes(cgroup_path: pathlib.Path) -> tuple[int, int]:
    pids: set[int] = set()
    for procs in cgroup_path.rglob("cgroup.procs"):
        try:
            pids.update(int(pid) for pid in procs.read_text().splitlines() if pid.strip())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    active = 0
    chrome = 0
    for pid in pids:
        try:
            command = (
                pathlib.Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .lower()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        active += "browseruse_bench/runner/agent_runner.py" in command
        executable = command.split(" ", 1)[0]
        chrome += any(token in executable for token in ("chrome", "chromium"))
    return active, chrome


class PssSampler:
    def __init__(self, cgroup_path: pathlib.Path):
        self.cgroup_path = cgroup_path
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest: tuple[float, float, ProcessTreeStats] | None = None
        self.total_failures = 0
        self.consecutive_failures = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                stats = summarize_processes(read_cgroup_processes(self.cgroup_path))
            except (FileNotFoundError, PermissionError, OSError):
                with self.lock:
                    self.total_failures += 1
                    self.consecutive_failures += 1
            else:
                finished = time.monotonic()
                with self.lock:
                    self.latest = (finished, finished - started, stats)
                    self.consecutive_failures = 0
            self.stop_event.wait(PSS_INTERVAL_SECONDS)

    def start(self) -> None:
        self.thread.start()

    def get(self) -> tuple[float, float, ProcessTreeStats] | None:
        with self.lock:
            return self.latest

    def health(self) -> tuple[int, int]:
        with self.lock:
            return self.total_failures, self.consecutive_failures

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)


def count_instances(run_dirs: Sequence[pathlib.Path]) -> tuple[int, int, int]:
    started = 0
    terminal = 0
    invalid = 0
    for run_dir in run_dirs:
        tasks = run_dir / "tasks"
        if not tasks.is_dir():
            continue
        for task_dir in tasks.iterdir():
            if not task_dir.is_dir():
                continue
            started += 1
            result = task_dir / "result.json"
            if result.is_file():
                try:
                    payload = json.loads(result.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("result is not an object")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    invalid += 1
                else:
                    terminal += 1
    return started, terminal, invalid


def reserve_timestamps(experiment_root: pathlib.Path, count: int) -> list[str]:
    experiment_root.mkdir(parents=True, exist_ok=True)
    timestamps: list[str] = []
    candidate = datetime.now(UTC).replace(microsecond=0)
    while len(timestamps) < count:
        timestamp = candidate.strftime("%Y%m%d_%H%M%S")
        path = experiment_root / timestamp
        try:
            path.mkdir(parents=False)
        except FileExistsError:
            candidate += timedelta(seconds=1)
            continue
        timestamps.append(timestamp)
        candidate += timedelta(seconds=1)
    return timestamps


def build_probe_manifest(backend: str, target: int, timestamps: Sequence[str]) -> dict[str, Any]:
    if target in TARGETS:
        return build_stress_manifest(backend, target, timestamps)
    if target <= 0 or target % 20 or len(timestamps) != target // 20:
        raise ValueError("Capacity probes must use complete 20-task replicas")
    return {
        "schema_version": 1,
        "capacity_probe": True,
        "seed": RANDOM_SEED,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_ids_sha256": EXPECTED_TASK_IDS_SHA256,
        "backend": backend,
        "target_concurrency": target,
        "per_replica_concurrency": 20,
        "replica_count": target // 20,
        "replicas": [
            {"index": index, "timestamp": timestamp} for index, timestamp in enumerate(timestamps)
        ],
    }


def _replica_command(manifest: dict[str, Any], index: int) -> list[str]:
    replica = manifest["replicas"][index]
    return build_official_replica_command(
        manifest["backend"], replica["timestamp"], manifest["task_ids"]
    )


def replica_worker(args: argparse.Namespace) -> int:
    manifest = json.loads(args.cell_manifest.read_text(encoding="utf-8"))
    ready = args.ready
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(f"{os.getpid()}\n", encoding="utf-8")
    deadline = time.monotonic() + 120
    while not args.barrier.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Replica launch barrier timed out")
        time.sleep(0.01)
    command = _replica_command(manifest, args.replica_index)
    os.chdir(manifest["checkout"])
    os.execvpe(command[0], command, os.environ)
    return 127


def cell_worker(args: argparse.Namespace) -> int:
    manifest = json.loads(args.cell_manifest.read_text(encoding="utf-8"))
    cell_root = pathlib.Path(manifest["cell_root"])
    barrier = cell_root / "launch.barrier"
    barrier.unlink(missing_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    stopping = False

    def stop_children(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        for _, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    for replica in manifest["replicas"]:
        index = int(replica["index"])
        replica_root = cell_root / "replicas" / f"r{index:02d}"
        replica_root.mkdir(parents=True, exist_ok=True)
        stdout = (replica_root / "official.stdout.log").open("wb")
        stderr = (replica_root / "official.stderr.log").open("wb")
        command = [
            sys.executable,
            "-m",
            MODULE_NAME,
            "replica-worker",
            "--cell-manifest",
            str(args.cell_manifest),
            "--replica-index",
            str(index),
            "--barrier",
            str(barrier),
            "--ready",
            str(replica_root / "ready"),
        ]
        process = subprocess.Popen(
            command,
            cwd=manifest["checkout"],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        processes.append((index, process))

    ready_paths = [
        cell_root / "replicas" / f"r{int(replica['index']):02d}" / "ready"
        for replica in manifest["replicas"]
    ]
    deadline = time.monotonic() + 60
    while not all(path.exists() for path in ready_paths):
        if stopping:
            return 143
        if time.monotonic() >= deadline:
            raise RuntimeError("Not all replicas reached the launch barrier")
        if any(process.poll() is not None for _, process in processes):
            raise RuntimeError("Replica worker exited before the launch barrier")
        time.sleep(0.05)
    barrier.write_text(f"{utc_now()}\n", encoding="utf-8")

    statuses: dict[str, int] = {}
    for index, process in processes:
        statuses[str(index)] = process.wait()
    atomic_json(cell_root / "replica_exit_status.json", statuses)
    return 143 if stopping else 0


def _write_resource_row(
    writer: csv.DictWriter,
    *,
    manifest: dict[str, Any],
    phase: str,
    monotonic_seconds: float,
    sample_gap: float,
    sample_cost: float,
    qwen: dict[str, float] | None,
    host_memory: Any,
    cgroup_path: pathlib.Path | None = None,
    cgroup: Any = None,
    cpu_delta: int = 0,
    cpu_cores: float = 0.0,
    pss: tuple[float, float, ProcessTreeStats] | None = None,
    active: int = 0,
    started: int = 0,
    terminal: int = 0,
    completed_delta: int = 0,
    throughput: float = 0.0,
) -> None:
    row = {field: "" for field in RESOURCE_CSV_FIELDS}
    row.update(
        {
            "timestamp_utc": utc_now(),
            "monotonic_seconds": f"{monotonic_seconds:.6f}",
            "phase": phase,
            "backend": manifest["backend"],
            "target_concurrency": manifest["target_concurrency"],
            "replica_count": manifest["replica_count"],
            "cgroup_path": str(cgroup_path or ""),
            "sample_gap_seconds": f"{sample_gap:.6f}",
            "sample_cost_seconds": f"{sample_cost:.6f}",
            "agent_runner_active": active,
            "started_instance_count": started,
            "terminal_instance_count": terminal,
            "completed_delta": completed_delta,
            "throughput_60s_task_per_hour": f"{throughput:.6f}",
            "host_memory_available_bytes": host_memory.available_bytes,
            "host_swap_free_bytes": host_memory.swap_free_bytes,
        }
    )
    if qwen:
        row.update({key: f"{value:.6f}" for key, value in qwen.items()})
    if cgroup:
        row.update(
            {
                "cpu_usage_usec": cgroup.cpu_usage_usec,
                "cpu_user_usec": cgroup.cpu_user_usec,
                "cpu_system_usec": cgroup.cpu_system_usec,
                "cpu_delta_usec": cpu_delta,
                "cpu_cores": f"{cpu_cores:.6f}",
                "memory_current_bytes": cgroup.memory_current_bytes,
                "memory_peak_bytes": cgroup.memory_peak_bytes,
                "memory_high_bytes": cgroup.memory_high_bytes,
                "memory_max_bytes": cgroup.memory_max_bytes,
                "memory_events_high": cgroup.memory_events_high,
                "memory_events_oom": cgroup.memory_events_oom,
                "memory_events_oom_kill": cgroup.memory_events_oom_kill,
                "pids_current": cgroup.pids_current,
                "pids_peak": cgroup.pids_peak,
            }
        )
    if pss:
        sampled_at, _duration, stats = pss
        row.update(
            {
                "process_count": stats.process_count,
                "chrome_process_count": stats.chrome_process_count,
                "chrome_session_active": stats.chrome_session_count,
                "process_tree_pss_bytes": stats.process_tree_pss_bytes,
                "chrome_pss_bytes": stats.chrome_pss_bytes,
                "nonchrome_pss_bytes": stats.nonchrome_pss_bytes,
                "pss_sample_age_seconds": f"{max(0.0, monotonic_seconds - sampled_at):.6f}",
            }
        )
    writer.writerow(row)


def classify_capacity_failure(cell_root: pathlib.Path) -> str | None:
    patterns = (
        (re.compile(r"session.{0,40}(quota|limit)|too many.{0,20}session", re.I), "session_quota"),
        (re.compile(r"websocket|\bcdp\b", re.I), "cdp_or_websocket"),
        (
            re.compile(r"(create|start).{0,30}session|session.{0,30}(create|start)", re.I),
            "session_create",
        ),
        (re.compile(r"\b429\b|rate.?limit", re.I), "upstream_rate_limit"),
        (re.compile(r"connection refused|connect(?:ion)? error", re.I), "qwen_or_network"),
        (re.compile(r"timed?\s*out|timeout", re.I), "task_timeout"),
    )
    candidates = list((cell_root / "replicas").glob("r*/official.*.log"))
    candidates.extend(
        path for path in cell_root.glob("**/run.log") if "tasks_eval_result" not in str(path)
    )
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[-2_000_000:]
        except OSError:
            continue
        for pattern, category in patterns:
            if pattern.search(text):
                return category
    return None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def monitor_cell(
    *,
    unit: str,
    manifest: dict[str, Any],
    cell_root: pathlib.Path,
    qwen_url: str,
    memory_max: int,
) -> dict[str, Any]:
    resource_path = cell_root / "monitor" / "resource_samples.csv"
    gpu_path = cell_root / "monitor" / "gpu_samples.csv"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    run_dirs = [pathlib.Path(replica["official_run_dir"]) for replica in manifest["replicas"]]
    gpu_utilization: list[float] = []
    baseline_fingerprints: set[str] = set()
    baseline_non_qwen_compute_present = False
    baseline_qwen: list[dict[str, float]] = []

    with (
        resource_path.open("w", newline="", encoding="utf-8") as resource_handle,
        gpu_path.open("w", newline="", encoding="utf-8") as gpu_handle,
    ):
        resource_writer = csv.DictWriter(resource_handle, fieldnames=RESOURCE_CSV_FIELDS)
        gpu_writer = csv.DictWriter(gpu_handle, fieldnames=GPU_CSV_FIELDS)
        resource_writer.writeheader()
        gpu_writer.writeheader()
        previous = time.monotonic()
        for _ in range(BASELINE_SECONDS):
            started_at = time.monotonic()
            host_memory = read_host_memory()
            qwen = read_qwen_metrics(qwen_url)
            rows = gpu_rows("baseline")
            for row in rows:
                gpu_writer.writerow(row)
                gpu_utilization.append(float(row["gpu_sm_percent"]))
                baseline_fingerprints.add(str(row["compute_pid_fingerprint"]))
                baseline_non_qwen_compute_present = (
                    baseline_non_qwen_compute_present or int(row["non_qwen_compute_pid_count"]) > 0
                )
            baseline_qwen.append(qwen)
            _write_resource_row(
                resource_writer,
                manifest=manifest,
                phase="baseline",
                monotonic_seconds=started_at,
                sample_gap=started_at - previous,
                sample_cost=time.monotonic() - started_at,
                qwen=qwen,
                host_memory=host_memory,
            )
            resource_handle.flush()
            gpu_handle.flush()
            previous = started_at
            time.sleep(max(0.0, 1.0 - (time.monotonic() - started_at)))

        # One external Qwen request from the intern is an approved, fixed
        # background load. GPU utilization and token deltas are therefore
        # observations only; the launch gate checks API reachability and that
        # the between-cell background queue never exceeds concurrency one.
        baseline_valid = all(
            sample["qwen_requests_running"] + sample["qwen_requests_waiting"] <= 1
            for sample in baseline_qwen
        )
        if not baseline_valid:
            raise RuntimeError("Qwen background concurrency exceeded one")
        if read_host_memory().available_bytes < HOST_MEMORY_RESERVE_BYTES + memory_max:
            raise RuntimeError("Safe host memory budget changed during the baseline")

        command = [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--no-block",
            "--property=Type=exec",
            "--property=CPUAccounting=yes",
            "--property=MemoryAccounting=yes",
            "--property=TasksAccounting=yes",
            f"--property=MemoryHigh={int(memory_max * MEMORY_HIGH_FRACTION)}",
            f"--property=MemoryMax={memory_max}",
            "--property=MemorySwapMax=0",
            f"--property=TasksMax={TASKS_MAX}",
            "--property=OOMPolicy=stop",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=30s",
            "--property=RuntimeMaxSec=1800s",
            f"--property=StandardOutput=append:{cell_root / 'service.stdout.log'}",
            f"--property=StandardError=append:{cell_root / 'service.stderr.log'}",
            "/bin/bash",
            "-lc",
            (
                "set -a; source "
                + shlex.quote(str(manifest["runtime_env"]))
                + '; set +a; export PATH="$HOME/.local/bin:$PATH"; exec '
                + shlex.quote(sys.executable)
                + " -m "
                + shlex.quote(MODULE_NAME)
                + " cell-worker --cell-manifest "
                + shlex.quote(str(cell_root / "cell_manifest.json"))
            ),
        ]
        subprocess.run(command, check=True, capture_output=True)
        append_event(cell_root / "monitor" / "events.jsonl", "service_started", unit=unit)

        cgroup_path: pathlib.Path | None = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            properties = systemctl_properties(unit)
            control_group = properties.get("ControlGroup", "")
            if control_group:
                candidate = pathlib.Path("/sys/fs/cgroup") / control_group.lstrip("/")
                if candidate.is_dir():
                    cgroup_path = candidate
                    break
            time.sleep(0.2)
        if cgroup_path is None:
            subprocess.run(["systemctl", "--user", "stop", unit], check=False)
            raise RuntimeError("Transient service cgroup was not created")

        cgroup_gate = read_cgroup_stats(cgroup_path)
        if cgroup_gate.memory_max_bytes != memory_max or cgroup_gate.pids_max != TASKS_MAX:
            subprocess.run(["systemctl", "--user", "stop", unit], check=False)
            raise RuntimeError("Transient service resource limits do not match the manifest")

        pss_sampler = PssSampler(cgroup_path)
        pss_sampler.start()
        previous_time = time.monotonic()
        previous_cpu = cgroup_gate.cpu_usage_usec
        previous_terminal = 0
        terminal_history: deque[tuple[float, int]] = deque()
        peak_active = 0
        reached_target = False
        sustained_seconds = 0.0
        max_sustained_seconds = 0.0
        natural_drain_after_reach = False
        memory_near_limit_samples = 0
        qwen_unreachable_samples = 0
        safety_reason: str | None = None
        rollout_fingerprints: set[str] = set()
        rollout_non_qwen_compute_present = False
        first_active_at: float | None = None
        service_end_at: float | None = None
        invalid_results = 0

        try:
            while True:
                sample_started = time.monotonic()
                properties = systemctl_properties(unit)
                state = properties.get("ActiveState", "unknown")
                try:
                    cgroup = read_cgroup_stats(cgroup_path)
                except FileNotFoundError:
                    if state not in {"active", "activating"}:
                        break
                    raise
                host_memory = read_host_memory()
                try:
                    qwen = read_qwen_metrics(qwen_url)
                    qwen_unreachable_samples = 0
                except Exception:  # noqa: BLE001
                    qwen = None
                    qwen_unreachable_samples += 1
                active, _chrome_processes = cheap_active_processes(cgroup_path)
                started, terminal, invalid_results = count_instances(run_dirs)
                peak_active = max(peak_active, active)
                if active and first_active_at is None:
                    first_active_at = sample_started
                threshold = math.ceil(0.95 * int(manifest["target_concurrency"]))
                reached_target = reached_target or active >= threshold
                if active >= threshold:
                    sustained_seconds += max(0.0, sample_started - previous_time)
                    max_sustained_seconds = max(max_sustained_seconds, sustained_seconds)
                else:
                    sustained_seconds = 0.0
                if reached_target and terminal >= math.ceil(
                    0.05 * int(manifest["target_concurrency"])
                ):
                    natural_drain_after_reach = True
                phase = "service_start"
                if active:
                    if active >= threshold:
                        phase = "steady"
                    else:
                        phase = "drain" if reached_target else "ramp"
                elif started:
                    phase = "drain"
                gap = sample_started - previous_time
                cpu_delta = cgroup.cpu_usage_usec - previous_cpu
                cores = calculate_cpu_cores(previous_cpu, cgroup.cpu_usage_usec, gap)
                completed_delta = terminal - previous_terminal
                terminal_history.append((sample_started, terminal))
                while terminal_history and sample_started - terminal_history[0][0] > 60:
                    terminal_history.popleft()
                throughput = 0.0
                if len(terminal_history) >= 2:
                    completed = terminal - terminal_history[0][1]
                    elapsed = sample_started - terminal_history[0][0]
                    if elapsed > 0:
                        throughput = calculate_throughput_per_hour(completed, elapsed)

                rows = gpu_rows(phase)
                for row in rows:
                    gpu_writer.writerow(row)
                    rollout_fingerprints.add(str(row["compute_pid_fingerprint"]))
                    rollout_non_qwen_compute_present = (
                        rollout_non_qwen_compute_present
                        or int(row["non_qwen_compute_pid_count"]) > 0
                    )
                _write_resource_row(
                    resource_writer,
                    manifest=manifest,
                    phase=phase,
                    monotonic_seconds=sample_started,
                    sample_gap=gap,
                    sample_cost=time.monotonic() - sample_started,
                    qwen=qwen,
                    host_memory=host_memory,
                    cgroup_path=cgroup_path,
                    cgroup=cgroup,
                    cpu_delta=cpu_delta,
                    cpu_cores=cores,
                    pss=pss_sampler.get(),
                    active=active,
                    started=started,
                    terminal=terminal,
                    completed_delta=completed_delta,
                    throughput=throughput,
                )
                resource_handle.flush()
                gpu_handle.flush()

                if cgroup.memory_events_oom_kill > 0:
                    safety_reason = "cgroup_oom_kill"
                elif cgroup.memory_events_oom > 0:
                    safety_reason = "cgroup_oom_allocation_failure"
                if cgroup.memory_current_bytes >= int(memory_max * MEMORY_STOP_FRACTION):
                    memory_near_limit_samples += 1
                else:
                    memory_near_limit_samples = 0
                if memory_near_limit_samples >= 3:
                    safety_reason = "memory_guard_90_percent"
                if host_memory.below_available_reserve(HOST_MEMORY_RESERVE_BYTES):
                    safety_reason = "host_memory_reserve"
                if cgroup.pids_max and cgroup.pids_current >= int(cgroup.pids_max * 0.9):
                    safety_reason = "pids_guard_90_percent"
                if gap > 15:
                    safety_reason = "sidecar_gap_over_15_seconds"
                if qwen_unreachable_samples >= 2:
                    safety_reason = "qwen_metrics_unreachable"
                pss_total_failures, pss_consecutive_failures = pss_sampler.health()
                latest_pss = pss_sampler.get()
                if pss_consecutive_failures >= 3:
                    safety_reason = "pss_sampling_failed"
                if latest_pss and sample_started - latest_pss[0] > 15:
                    safety_reason = "pss_sample_stale"
                if safety_reason:
                    append_event(
                        cell_root / "monitor" / "events.jsonl",
                        "safety_stop",
                        reason=safety_reason,
                    )
                    subprocess.run(["systemctl", "--user", "stop", unit], check=False)

                previous_time = sample_started
                service_end_at = sample_started
                previous_cpu = cgroup.cpu_usage_usec
                previous_terminal = terminal
                if state not in {"active", "activating"}:
                    break
                time.sleep(max(0.0, 1.0 - (time.monotonic() - sample_started)))
        finally:
            pss_sampler.stop()

    properties = systemctl_properties(unit)
    started, terminal, invalid_results = count_instances(run_dirs)
    protocol_errors: list[str] = []
    for replica in manifest["replicas"]:
        snapshot_path = pathlib.Path(replica["official_run_dir"]) / "config_snapshot.json"
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            validate_config_snapshot(
                snapshot,
                backend=manifest["backend"],
                timestamp=replica["timestamp"],
                task_ids=manifest["task_ids"],
            )
        except Exception as exc:  # noqa: BLE001
            protocol_errors.append(f"replica={replica['index']}: {exc}")
    required_active = math.ceil(0.95 * int(manifest["target_concurrency"]))
    required_results = math.ceil(0.95 * int(manifest["target_concurrency"]))
    stability_pass = max_sustained_seconds >= 60.0 or natural_drain_after_reach
    exit_status_path = cell_root / "replica_exit_status.json"
    replica_exit_statuses = (
        json.loads(exit_status_path.read_text(encoding="utf-8"))
        if exit_status_path.is_file()
        else {}
    )
    nonzero_replicas = sum(int(status) != 0 for status in replica_exit_statuses.values())
    capacity_pass = (
        safety_reason is None
        and not protocol_errors
        and peak_active >= required_active
        and stability_pass
        and terminal >= required_results
        and invalid_results == 0
    )
    rollout_seconds = 0.0
    if first_active_at is not None and service_end_at is not None:
        rollout_seconds = max(0.0, service_end_at - first_active_at)
    capacity_failure_reason: str | None = None
    if not capacity_pass:
        capacity_failure_reason = safety_reason or classify_capacity_failure(cell_root)
        if capacity_failure_reason is None and protocol_errors:
            capacity_failure_reason = "protocol_mismatch"
        elif capacity_failure_reason is None and peak_active < required_active:
            capacity_failure_reason = "actual_concurrency_below_95_percent"
        elif capacity_failure_reason is None and not stability_pass:
            capacity_failure_reason = "concurrency_not_sustained"
        elif capacity_failure_reason is None and terminal < required_results:
            capacity_failure_reason = "result_coverage_below_95_percent"
        elif capacity_failure_reason is None and invalid_results:
            capacity_failure_reason = "invalid_result_json"
    return {
        "schema_version": 1,
        "backend": manifest["backend"],
        "target_concurrency": manifest["target_concurrency"],
        "replica_count": manifest["replica_count"],
        "planned_instances": manifest["target_concurrency"],
        "started_instances": started,
        "terminal_instances": terminal,
        "invalid_result_files": invalid_results,
        "peak_active": peak_active,
        "max_sustained_seconds_at_95_percent": max_sustained_seconds,
        "natural_drain_after_reach": natural_drain_after_reach,
        "stability_pass": stability_pass,
        "required_active_95_percent": required_active,
        "required_results_95_percent": required_results,
        "capacity_pass": capacity_pass,
        "safety_reason": safety_reason,
        "capacity_failure_reason": capacity_failure_reason,
        "protocol_errors": protocol_errors,
        "replica_exit_statuses": replica_exit_statuses,
        "nonzero_replica_count": nonzero_replicas,
        "systemd_result": properties.get("Result"),
        "systemd_exec_main_status": properties.get("ExecMainStatus"),
        "rollout_seconds": rollout_seconds,
        "throughput_task_per_hour": (
            calculate_throughput_per_hour(terminal, rollout_seconds) if rollout_seconds > 0 else 0.0
        ),
        "gpu_non_qwen_pid_set_stable": (
            len(rollout_fingerprints) == 1 and rollout_fingerprints == baseline_fingerprints
        ),
        "gpu_metric_scope": "whole_device_baseline_delta",
        "approved_external_qwen_concurrency": 1,
        "gpu_idle_is_launch_gate": False,
        "gpu_process_attribution_valid": not (
            baseline_non_qwen_compute_present or rollout_non_qwen_compute_present
        ),
        "gpu_limitation": (
            "non_qwen_compute_process_present"
            if baseline_non_qwen_compute_present or rollout_non_qwen_compute_present
            else None
        ),
        "pss_sampling_failures": pss_sampler.health()[0],
        "official_run_dirs": [str(path) for path in run_dirs],
        "completed_at": utc_now(),
    }


def _preflight_config(checkout: pathlib.Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError("Official checkout commit drift")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    unexpected = [line for line in status if line[3:] != "config.yaml"]
    if unexpected:
        raise RuntimeError("Official checkout contains unexpected tracked changes")
    dataset = checkout / "browseruse_bench/data/LexBench-Browser/task.jsonl"
    validate_frozen_sample(dataset.read_bytes())
    import yaml  # Imported only on the Linux runner where PyYAML is installed.

    config = yaml.safe_load((checkout / "config.yaml").read_text(encoding="utf-8"))
    model = config["models"]["qwen3-8B"]
    agent = config["agents"]["browser-use"]
    expected = {
        "model_id": model.get("model_id") == "$QWEN_MODEL_ID",
        "structured": model.get("dont_force_structured_output") is False,
        "schema": model.get("add_schema_to_system_prompt") is True,
        "max_steps": agent.get("max_steps") == 40,
        "timeout": agent.get("timeout") == 600,
        "flash": agent.get("flash_mode") is True,
        "vision": agent.get("use_vision") is False,
        "judge": agent.get("use_judge") is False,
        "backends": all(name in config["browsers"] for name in ("lexmount", "local")),
    }
    if not all(expected.values()):
        raise RuntimeError(f"Official config protocol drift: {expected}")


def prepare_cell(
    *,
    campaign_id: str,
    campaign_root: pathlib.Path,
    checkout: pathlib.Path,
    runtime_env: pathlib.Path,
    backend: str,
    target: int,
    name: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    cell_root = campaign_root / "cells" / name
    if (cell_root / "cell_summary.json").is_file():
        return cell_root, json.loads((cell_root / "cell_manifest.json").read_text())
    if cell_root.exists() and any(cell_root.iterdir()):
        raise RuntimeError(f"Incomplete cell already exists: {cell_root}")
    cell_root.mkdir(parents=True, exist_ok=True)
    experiment_root = checkout / "experiments/LexBench-Browser/All/browser-use/qwen3_8B"
    timestamps = reserve_timestamps(experiment_root, target // 20)
    manifest = build_probe_manifest(backend, target, timestamps)
    manifest.update(
        {
            "campaign_id": campaign_id,
            "cell_name": name,
            "cell_root": str(cell_root),
            "checkout": str(checkout),
            "runtime_env": str(runtime_env),
            "created_at": utc_now(),
        }
    )
    for replica in manifest["replicas"]:
        replica["official_run_dir"] = str(experiment_root / replica["timestamp"])
    manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
    atomic_json(cell_root / "cell_manifest.json", manifest)
    return cell_root, manifest


def run_one_cell(
    *,
    campaign_id: str,
    campaign_root: pathlib.Path,
    checkout: pathlib.Path,
    runtime_env: pathlib.Path,
    qwen_url: str,
    memory_max: int,
    backend: str,
    target: int,
    name: str,
) -> dict[str, Any]:
    cell_root, manifest = prepare_cell(
        campaign_id=campaign_id,
        campaign_root=campaign_root,
        checkout=checkout,
        runtime_env=runtime_env,
        backend=backend,
        target=target,
        name=name,
    )
    summary_path = cell_root / "cell_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    (campaign_root / "current_cell.txt").write_text(f"{name}\n", encoding="utf-8")
    append_event(campaign_root / "campaign_events.jsonl", "cell_start", name=name)
    attempt_id = str(manifest["replicas"][0]["timestamp"])
    unit_base = re.sub(r"[^a-zA-Z0-9-]", "-", f"{campaign_id}-{name}-{attempt_id}")[-180:]
    unit = f"{unit_base}.service"
    try:
        summary = monitor_cell(
            unit=unit,
            manifest=manifest,
            cell_root=cell_root,
            qwen_url=qwen_url,
            memory_max=memory_max,
        )
    except BaseException:
        subprocess.run(
            ["systemctl", "--user", "stop", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if systemctl_properties(unit).get("ActiveState") not in {"active", "activating"}:
                    break
            except subprocess.CalledProcessError:
                break
            time.sleep(0.5)
        raise
    atomic_json(summary_path, summary)
    append_event(
        campaign_root / "campaign_events.jsonl",
        "cell_complete",
        name=name,
        capacity_pass=summary["capacity_pass"],
    )
    time.sleep(COOLDOWN_SECONDS)
    return summary


def binary_targets(low_pass: int, high_fail: int) -> list[int]:
    probes: list[int] = []
    low = low_pass
    high = high_fail
    while high - low > 20:
        mid = ((low + high) // 40) * 20
        if mid <= low:
            mid = low + 20
        if mid >= high:
            break
        probes.append(mid)
        # The caller updates the interval after each observed result.
        break
    return probes


def campaign(args: argparse.Namespace) -> int:
    runtime_root = args.runtime_root.resolve()
    checkout = runtime_root / ".lexbench/browseruse-agent-bench"
    runtime_env = args.env_file.resolve()
    campaign_root = (
        runtime_root / "results/lexbench" / args.campaign_id / "stress_process_attributed"
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    _preflight_config(checkout)
    env_values = read_runtime_env(runtime_env)
    qwen_url = qwen_metrics_url(env_values["QWEN_BASE_URL"])
    host_memory = read_host_memory()
    memory_max = min(
        MAX_CELL_MEMORY_BYTES,
        host_memory.available_bytes - HOST_MEMORY_RESERVE_BYTES - MEMORY_BUDGET_HEADROOM_BYTES,
    )
    if memory_max < MIN_CELL_MEMORY_BYTES:
        raise RuntimeError("Insufficient safe memory budget for the first stress cell")
    campaign_manifest = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "created_at": utc_now(),
        "official_commit": OFFICIAL_COMMIT,
        "seed": RANDOM_SEED,
        "task_ids": list(EXPECTED_TASK_IDS),
        "targets": list(TARGETS),
        "replicas_per_target": {str(target): target // 20 for target in TARGETS},
        "memory_max_bytes": memory_max,
        "memory_high_bytes": int(memory_max * MEMORY_HIGH_FRACTION),
        "host_memory_reserve_bytes": HOST_MEMORY_RESERVE_BYTES,
        "tasks_max": TASKS_MAX,
        "stage2_judge_model": "gpt-5.4",
        "stage2_concurrency": 5,
        "backends": list(args.backends),
        "gpu_metric_scope": "whole_device_baseline_delta",
        "approved_external_qwen_concurrency": 1,
        "gpu_idle_is_launch_gate": False,
        "gpu_process_attribution_requires_no_non_qwen_compute_process": True,
    }
    atomic_json(campaign_root / "campaign_manifest.json", campaign_manifest)
    append_event(campaign_root / "campaign_events.jsonl", "campaign_start")

    history: dict[str, list[dict[str, Any]]] = {backend: [] for backend in args.backends}
    stopped: dict[str, bool] = {backend: False for backend in args.backends}
    first_failure: dict[str, int | None] = {backend: None for backend in args.backends}
    last_pass: dict[str, int] = {backend: 0 for backend in args.backends}

    for target_index, target in enumerate(TARGETS):
        preferred = ("lexmount", "local") if target_index % 2 == 0 else ("local", "lexmount")
        order = tuple(backend for backend in preferred if backend in args.backends)
        for backend in order:
            if stopped[backend]:
                continue
            name = f"{backend}_c{target}"
            summary = run_one_cell(
                campaign_id=args.campaign_id,
                campaign_root=campaign_root,
                checkout=checkout,
                runtime_env=runtime_env,
                qwen_url=qwen_url,
                memory_max=memory_max,
                backend=backend,
                target=target,
                name=name,
            )
            history[backend].append(summary)
            if summary["capacity_pass"]:
                last_pass[backend] = target
            else:
                first_failure[backend] = target
                stopped[backend] = True

    for backend in args.backends:
        high = first_failure[backend]
        low = last_pass[backend]
        while high is not None and high - low > 20:
            candidates = binary_targets(low, high)
            if not candidates:
                break
            target = candidates[0]
            summary = run_one_cell(
                campaign_id=args.campaign_id,
                campaign_root=campaign_root,
                checkout=checkout,
                runtime_env=runtime_env,
                qwen_url=qwen_url,
                memory_max=memory_max,
                backend=backend,
                target=target,
                name=f"{backend}_capacity_probe_c{target}",
            )
            history[backend].append(summary)
            if summary["capacity_pass"]:
                low = target
            else:
                high = target
        last_pass[backend] = low

    campaign_summary = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "rollout_complete": True,
        "stage2_complete": False,
        "maximum_sustainable_concurrency": last_pass,
        "first_failed_concurrency": first_failure,
        "cells": history,
    }
    atomic_json(campaign_root / "rollout_summary.json", campaign_summary)
    (campaign_root / "current_cell.txt").unlink(missing_ok=True)
    append_event(campaign_root / "campaign_events.jsonl", "rollout_campaign_complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    campaign_parser = subparsers.add_parser("campaign")
    campaign_parser.add_argument("--campaign-id", required=True)
    campaign_parser.add_argument(
        "--runtime-root", type=pathlib.Path, default=pathlib.Path("/data/wf/sxh")
    )
    campaign_parser.add_argument("--env-file", type=pathlib.Path, required=True)
    campaign_parser.add_argument("--backend", choices=("lexmount", "local", "all"), default="all")
    worker = subparsers.add_parser("cell-worker")
    worker.add_argument("--cell-manifest", type=pathlib.Path, required=True)
    replica = subparsers.add_parser("replica-worker")
    replica.add_argument("--cell-manifest", type=pathlib.Path, required=True)
    replica.add_argument("--replica-index", type=int, required=True)
    replica.add_argument("--barrier", type=pathlib.Path, required=True)
    replica.add_argument("--ready", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "campaign":
        args.backends = ("lexmount", "local") if args.backend == "all" else (args.backend,)
        return campaign(args)
    if args.command == "cell-worker":
        return cell_worker(args)
    if args.command == "replica-worker":
        return replica_worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
