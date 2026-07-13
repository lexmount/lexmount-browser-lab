from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import signal
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

GPU_QUERY = "index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
CSV_FIELDS = (
    "timestamp_utc",
    "monotonic_seconds",
    "phase",
    "root_pid",
    "sample_gap_seconds",
    "host_cpu_percent",
    "host_memory_used_bytes",
    "host_memory_available_bytes",
    "swap_used_bytes",
    "runner_cpu_seconds",
    "runner_rss_bytes",
    "browser_cpu_seconds",
    "browser_rss_bytes",
    "vllm_cpu_seconds",
    "vllm_rss_bytes",
    "gpu_index",
    "gpu_uuid",
    "gpu_sm_percent",
    "gpu_memory_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_power_w",
    "terminal_task_count",
)


@dataclass(frozen=True)
class MonitorConfig:
    output_dir: pathlib.Path
    cwd: pathlib.Path
    experiment_root: pathlib.Path
    expected_tasks: int
    has_judge: bool
    existing_run_dir: pathlib.Path | None = None
    interval_seconds: float = 1.0
    baseline_seconds: int = 30


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    name: str
    command: str
    cpu_seconds: float
    rss_bytes: int


def parse_nvidia_rows(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in output.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 7:
            raise RuntimeError("Malformed NVIDIA sample")
        try:
            rows.append(
                {
                    "gpu_index": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "gpu_sm_percent": float(fields[2]),
                    "gpu_memory_percent": float(fields[3]),
                    "gpu_memory_used_mib": float(fields[4]),
                    "gpu_memory_total_mib": float(fields[5]),
                    "gpu_power_w": float(fields[6]),
                }
            )
        except ValueError:
            raise RuntimeError("Malformed NVIDIA sample") from None
    if not rows:
        raise RuntimeError("Empty NVIDIA sample")
    return rows


def classify_local_browser(name: str, command: str) -> bool:
    lowered_name = name.lower()
    lowered_command = command.lower()
    names = ("chrome", "chromium", "chromium-browser", "google-chrome")
    return any(token in lowered_name for token in names) or any(
        f"/{token}" in lowered_command for token in names
    )


def discover_new_run_dir(root: pathlib.Path, before: set[str]) -> pathlib.Path:
    current = {path.name for path in root.iterdir() if path.is_dir()}
    created = sorted(current - before)
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new official run directory, found {len(created)}")
    return root / created[0]


def phase_for(has_judge: bool, terminal_tasks: int, expected_tasks: int) -> str:
    if has_judge and terminal_tasks >= expected_tasks:
        return "judge"
    return "rollout"


def _read_processes() -> dict[int, ProcessRecord]:
    records: dict[int, ProcessRecord] = {}
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    page_size = os.sysconf("SC_PAGE_SIZE")
    for path in pathlib.Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text(encoding="utf-8").split()
            command = (
                (path / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
            records[int(path.name)] = ProcessRecord(
                pid=int(path.name),
                ppid=int(stat[3]),
                name=stat[1].strip("()"),
                command=command,
                cpu_seconds=(int(stat[13]) + int(stat[14])) / ticks,
                rss_bytes=int(stat[23]) * page_size,
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return records


def _descendants(records: dict[int, ProcessRecord], root_pid: int) -> set[int]:
    result = {root_pid}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [record.pid for record in records.values() if record.ppid == parent]
        for child in children:
            if child not in result:
                result.add(child)
                frontier.append(child)
    return result


def _totals(records: dict[int, ProcessRecord], pids: Iterable[int]) -> tuple[float, int]:
    selected = [records[pid] for pid in pids if pid in records]
    return (
        sum(record.cpu_seconds for record in selected),
        sum(record.rss_bytes for record in selected),
    )


def _read_cpu_counters() -> tuple[int, int]:
    fields = pathlib.Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int]) -> float:
    if previous is None:
        return 0.0
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return 0.0
    return 100.0 * (total_delta - idle_delta) / total_delta


def _read_memory() -> tuple[int, int, int]:
    values: dict[str, int] = {}
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.strip().split()[0]) * 1024
    total = values["MemTotal"]
    available = values["MemAvailable"]
    swap_used = values["SwapTotal"] - values["SwapFree"]
    return total - available, available, swap_used


def _gpu_rows() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_nvidia_rows(completed.stdout)


def _terminal_count(run_dir: pathlib.Path | None) -> int:
    if run_dir is None:
        return 0
    return sum(1 for _ in (run_dir / "tasks").glob("*/result.json"))


class ResourceSidecar:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self._last_cpu: tuple[int, int] | None = None
        self._last_monotonic: float | None = None

    def _event(self, handle, name: str, **fields: object) -> None:
        payload = {"timestamp_utc": _utc_now(), "event": name, **fields}
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        handle.flush()

    def _sample(
        self,
        writer: csv.DictWriter,
        phase: str,
        root_pid: int,
        run_dir: pathlib.Path | None,
    ) -> None:
        now = time.monotonic()
        gap = 0.0 if self._last_monotonic is None else now - self._last_monotonic
        self._last_monotonic = now
        counters = _read_cpu_counters()
        host_cpu = _cpu_percent(self._last_cpu, counters)
        self._last_cpu = counters
        memory_used, memory_available, swap_used = _read_memory()
        records = _read_processes()
        runner_pids = _descendants(records, root_pid) if root_pid else set()
        runner_cpu, runner_rss = _totals(records, runner_pids)
        browser_pids = {
            pid
            for pid in runner_pids
            if pid in records and classify_local_browser(records[pid].name, records[pid].command)
        }
        browser_cpu, browser_rss = _totals(records, browser_pids)
        vllm_pids = {
            record.pid
            for record in records.values()
            if "vllm" in record.command.lower() or "qwen3-8b" in record.command.lower()
        }
        vllm_cpu, vllm_rss = _totals(records, vllm_pids)
        terminal_tasks = _terminal_count(run_dir)
        base = {
            "timestamp_utc": _utc_now(),
            "monotonic_seconds": f"{now:.6f}",
            "phase": phase,
            "root_pid": root_pid,
            "sample_gap_seconds": f"{gap:.6f}",
            "host_cpu_percent": f"{host_cpu:.3f}",
            "host_memory_used_bytes": memory_used,
            "host_memory_available_bytes": memory_available,
            "swap_used_bytes": swap_used,
            "runner_cpu_seconds": f"{runner_cpu:.3f}",
            "runner_rss_bytes": runner_rss,
            "browser_cpu_seconds": f"{browser_cpu:.3f}",
            "browser_rss_bytes": browser_rss,
            "vllm_cpu_seconds": f"{vllm_cpu:.3f}",
            "vllm_rss_bytes": vllm_rss,
            "terminal_task_count": terminal_tasks,
        }
        for gpu in _gpu_rows():
            writer.writerow({**base, **gpu})

    def run(self, command: list[str], env: dict[str, str]) -> int:
        config = self.config
        if config.existing_run_dir is not None and not config.existing_run_dir.is_dir():
            raise RuntimeError(
                f"Existing official run directory does not exist: {config.existing_run_dir}"
            )
        config.output_dir.mkdir(parents=True, exist_ok=False)
        samples_path = config.output_dir / "resource_samples.csv"
        events_path = config.output_dir / "monitor_events.jsonl"
        stdout_path = config.output_dir / "official.stdout.log"
        stderr_path = config.output_dir / "official.stderr.log"
        before = {path.name for path in config.experiment_root.iterdir() if path.is_dir()}
        process: subprocess.Popen[bytes] | None = None
        run_dir = config.existing_run_dir

        with (
            samples_path.open("w", encoding="utf-8", newline="") as sample_handle,
            events_path.open("w", encoding="utf-8") as event_handle,
            stdout_path.open("wb") as stdout_handle,
            stderr_path.open("wb") as stderr_handle,
        ):
            writer = csv.DictWriter(sample_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            if run_dir is not None:
                self._event(
                    event_handle,
                    "official_run_bound",
                    path=str(run_dir),
                    resumed=True,
                    terminal_task_count=_terminal_count(run_dir),
                )
            self._event(event_handle, "baseline_start")
            baseline_samples = max(1, round(config.baseline_seconds / config.interval_seconds))
            for _ in range(baseline_samples):
                started = time.monotonic()
                self._sample(writer, "baseline", 0, None)
                sample_handle.flush()
                time.sleep(max(0.0, config.interval_seconds - (time.monotonic() - started)))
            self._event(event_handle, "baseline_complete", samples=baseline_samples)

            process = subprocess.Popen(
                command,
                cwd=config.cwd,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            self._event(event_handle, "runner_start", root_pid=process.pid)

            def stop_runner(signum, _frame) -> None:
                self._event(event_handle, "signal", signal=signum)
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signum)

            previous_term = signal.signal(signal.SIGTERM, stop_runner)
            previous_int = signal.signal(signal.SIGINT, stop_runner)
            try:
                while process.poll() is None:
                    started = time.monotonic()
                    if run_dir is None:
                        created = [
                            path
                            for path in config.experiment_root.iterdir()
                            if path.is_dir() and path.name not in before
                        ]
                        if len(created) == 1:
                            run_dir = created[0]
                            self._event(event_handle, "official_run_bound", path=str(run_dir))
                        elif len(created) > 1:
                            raise RuntimeError("Expected exactly one new official run directory")
                    terminal_tasks = _terminal_count(run_dir)
                    phase = phase_for(config.has_judge, terminal_tasks, config.expected_tasks)
                    self._sample(writer, phase, process.pid, run_dir)
                    sample_handle.flush()
                    time.sleep(max(0.0, config.interval_seconds - (time.monotonic() - started)))
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)

            status = process.wait()
            if run_dir is None:
                run_dir = discover_new_run_dir(config.experiment_root, before)
                self._event(event_handle, "official_run_bound", path=str(run_dir))
            self._event(
                event_handle,
                "runner_exit",
                root_pid=process.pid,
                status=status,
                terminal_task_count=_terminal_count(run_dir),
            )
            return status


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor one official LexBench command")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cwd", type=pathlib.Path, required=True)
    parser.add_argument("--experiment-root", type=pathlib.Path, required=True)
    parser.add_argument("--existing-run-dir", type=pathlib.Path)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--has-judge", action="store_true")
    parser.add_argument("--baseline-seconds", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("official command is required after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = MonitorConfig(
        output_dir=args.output_dir,
        cwd=args.cwd,
        experiment_root=args.experiment_root,
        expected_tasks=args.expected_tasks,
        has_judge=args.has_judge,
        existing_run_dir=args.existing_run_dir,
        interval_seconds=args.interval_seconds,
        baseline_seconds=args.baseline_seconds,
    )
    return ResourceSidecar(config).run(args.command, dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
