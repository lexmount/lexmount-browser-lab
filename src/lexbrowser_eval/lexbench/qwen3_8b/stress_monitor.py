"""Process-attributed resource sampling primitives for LexBench stress cells.

The module is intentionally side-effect free: callers provide cgroup and proc
paths, while orchestration, process launching, and CSV writing live elsewhere.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

RESOURCE_CSV_FIELDS = (
    "timestamp_utc",
    "monotonic_seconds",
    "phase",
    "backend",
    "target_concurrency",
    "replica_count",
    "cgroup_path",
    "sample_gap_seconds",
    "sample_cost_seconds",
    "cpu_usage_usec",
    "cpu_user_usec",
    "cpu_system_usec",
    "cpu_delta_usec",
    "cpu_cores",
    "memory_current_bytes",
    "memory_peak_bytes",
    "memory_high_bytes",
    "memory_max_bytes",
    "memory_events_high",
    "memory_events_oom",
    "memory_events_oom_kill",
    "pids_current",
    "pids_peak",
    "process_count",
    "agent_runner_active",
    "chrome_process_count",
    "chrome_session_active",
    "process_tree_pss_bytes",
    "chrome_pss_bytes",
    "nonchrome_pss_bytes",
    "pss_sample_age_seconds",
    "started_instance_count",
    "terminal_instance_count",
    "completed_delta",
    "throughput_60s_task_per_hour",
    "host_memory_available_bytes",
    "host_swap_free_bytes",
    "qwen_requests_running",
    "qwen_requests_waiting",
    "qwen_prompt_tokens_total",
    "qwen_generation_tokens_total",
)


@dataclass(frozen=True)
class CgroupStats:
    cpu_usage_usec: int
    cpu_user_usec: int
    cpu_system_usec: int
    memory_current_bytes: int
    memory_peak_bytes: int
    memory_high_bytes: int | None
    memory_max_bytes: int | None
    memory_events_high: int
    memory_events_oom: int
    memory_events_oom_kill: int
    pids_current: int
    pids_peak: int
    pids_max: int | None


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    ppid: int
    name: str
    command: str
    pss_bytes: int
    rss_bytes: int


@dataclass(frozen=True)
class ProcessTreeStats:
    process_count: int
    active_agent_runners: int
    chrome_process_count: int
    chrome_session_count: int
    process_tree_pss_bytes: int
    chrome_pss_bytes: int
    nonchrome_pss_bytes: int


@dataclass(frozen=True)
class HostMemoryStats:
    total_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int

    def below_available_reserve(self, reserve_bytes: int) -> bool:
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")
        return self.available_bytes < reserve_bytes


def _read_key_values(path: pathlib.Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(None, 1)
        values[key] = int(raw_value)
    return values


def _read_limit(path: pathlib.Path) -> int | None:
    value = path.read_text(encoding="utf-8").strip()
    return None if value == "max" else int(value)


def read_cgroup_stats(cgroup_path: pathlib.Path) -> CgroupStats:
    """Read one cgroup-v2 CPU, memory, OOM, and PID counter snapshot."""

    cpu = _read_key_values(cgroup_path / "cpu.stat")
    memory_events = _read_key_values(cgroup_path / "memory.events")
    return CgroupStats(
        cpu_usage_usec=cpu["usage_usec"],
        cpu_user_usec=cpu["user_usec"],
        cpu_system_usec=cpu["system_usec"],
        memory_current_bytes=int((cgroup_path / "memory.current").read_text(encoding="utf-8")),
        memory_peak_bytes=int((cgroup_path / "memory.peak").read_text(encoding="utf-8")),
        memory_high_bytes=_read_limit(cgroup_path / "memory.high"),
        memory_max_bytes=_read_limit(cgroup_path / "memory.max"),
        memory_events_high=memory_events["high"],
        memory_events_oom=memory_events["oom"],
        memory_events_oom_kill=memory_events["oom_kill"],
        pids_current=int((cgroup_path / "pids.current").read_text(encoding="utf-8")),
        pids_peak=int((cgroup_path / "pids.peak").read_text(encoding="utf-8")),
        pids_max=_read_limit(cgroup_path / "pids.max"),
    )


def _member_pids(cgroup_path: pathlib.Path) -> set[int]:
    pids: set[int] = set()
    for procs_file in cgroup_path.rglob("cgroup.procs"):
        for raw_pid in procs_file.read_text(encoding="utf-8").splitlines():
            if raw_pid.strip():
                pids.add(int(raw_pid))
    return pids


def _read_status_ppid(path: pathlib.Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PPid:"):
            return int(line.split(":", 1)[1])
    raise ValueError(f"PPid is missing from {path}")


def _read_smaps_rollup(path: pathlib.Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"Pss", "Rss"}:
            values[key] = int(raw.strip().split()[0]) * 1024
    return values["Pss"], values["Rss"]


def read_cgroup_processes(
    cgroup_path: pathlib.Path,
    *,
    proc_root: pathlib.Path = pathlib.Path("/proc"),
) -> list[ProcessSample]:
    """Read live process identity and proportional memory for a cgroup tree.

    Processes can exit between reading ``cgroup.procs`` and ``smaps_rollup``;
    such races are expected and omitted from this point-in-time snapshot.
    """

    samples: list[ProcessSample] = []
    for pid in sorted(_member_pids(cgroup_path)):
        process_dir = proc_root / str(pid)
        try:
            command = (
                (process_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
            pss_bytes, rss_bytes = _read_smaps_rollup(process_dir / "smaps_rollup")
            samples.append(
                ProcessSample(
                    pid=pid,
                    ppid=_read_status_ppid(process_dir / "status"),
                    name=(process_dir / "comm").read_text(encoding="utf-8").strip(),
                    command=command,
                    pss_bytes=pss_bytes,
                    rss_bytes=rss_bytes,
                )
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, KeyError):
            continue
    return samples


def is_agent_runner(process: ProcessSample) -> bool:
    return "browseruse_bench/runner/agent_runner.py" in process.command


def is_chrome_process(process: ProcessSample) -> bool:
    name = process.name.lower()
    executable = process.command.split(" ", 1)[0].lower()
    return (
        "chrome" in name
        or "chromium" in name
        or executable.endswith("/chrome")
        or executable.endswith("/chromium")
        or executable.endswith("/google-chrome")
    )


def _is_chrome_main_session(
    process: ProcessSample,
    by_pid: dict[int, ProcessSample],
) -> bool:
    lowered = process.command.lower()
    if not is_chrome_process(process) or "crashpad" in lowered or "--type=" in lowered:
        return False
    ancestor_pid = process.ppid
    seen: set[int] = set()
    while ancestor_pid in by_pid and ancestor_pid not in seen:
        seen.add(ancestor_pid)
        ancestor = by_pid[ancestor_pid]
        if is_chrome_process(ancestor):
            return False
        ancestor_pid = ancestor.ppid
    return True


def summarize_processes(processes: list[ProcessSample]) -> ProcessTreeStats:
    """Summarize active tasks, Local Chrome sessions, and PSS attribution."""

    by_pid = {process.pid: process for process in processes}
    chrome = [process for process in processes if is_chrome_process(process)]
    total_pss = sum(process.pss_bytes for process in processes)
    chrome_pss = sum(process.pss_bytes for process in chrome)
    return ProcessTreeStats(
        process_count=len(processes),
        active_agent_runners=sum(is_agent_runner(process) for process in processes),
        chrome_process_count=len(chrome),
        chrome_session_count=sum(_is_chrome_main_session(process, by_pid) for process in chrome),
        process_tree_pss_bytes=total_pss,
        chrome_pss_bytes=chrome_pss,
        nonchrome_pss_bytes=total_pss - chrome_pss,
    )


def read_host_memory(
    meminfo_path: pathlib.Path = pathlib.Path("/proc/meminfo"),
) -> HostMemoryStats:
    """Read only the host values required for memory safety guardrails."""

    values: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(raw.strip().split()[0]) * 1024
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing meminfo fields: {sorted(missing)}")
    return HostMemoryStats(
        total_bytes=values["MemTotal"],
        available_bytes=values["MemAvailable"],
        swap_total_bytes=values["SwapTotal"],
        swap_free_bytes=values["SwapFree"],
    )


def calculate_cpu_cores(
    previous_usage_usec: int,
    current_usage_usec: int,
    elapsed_seconds: float,
) -> float:
    """Convert cumulative cgroup CPU time into mean utilized CPU cores."""

    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    delta = current_usage_usec - previous_usage_usec
    if delta < 0:
        raise ValueError("cgroup CPU usage must be monotonic")
    return delta / (elapsed_seconds * 1_000_000)


def calculate_throughput_per_hour(completed_tasks: int, elapsed_seconds: float) -> float:
    """Calculate completed task throughput over a positive observation window."""

    if completed_tasks < 0:
        raise ValueError("completed_tasks must be non-negative")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return completed_tasks * 3600.0 / elapsed_seconds
