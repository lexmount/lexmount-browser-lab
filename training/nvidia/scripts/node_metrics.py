#!/usr/bin/env python3
"""Run one lightweight, detached resource sampler on each Ray node."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path


def sample(run_dir: str) -> dict[str, object]:
    cpu = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    mem = dict(
        line.split(":", 1)
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        if ":" in line
    )
    net = {}
    for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]:
        name, values = line.split(":", 1)
        parts = values.split()
        if name.strip() != "lo" and len(parts) >= 9:
            net[name.strip()] = {"rx_bytes": int(parts[0]), "tx_bytes": int(parts[8])}
    gpus = []
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 4:
                gpus.append(
                    {
                        "index": int(fields[0]),
                        "utilization_pct": float(fields[1]),
                        "memory_used_mib": float(fields[2]),
                        "memory_total_mib": float(fields[3]),
                    }
                )
    disk = shutil.disk_usage(run_dir)
    return {
        "timestamp_unix": time.time(),
        "hostname": socket.gethostname(),
        "cpu_ticks": [int(value) for value in cpu],
        "memory_total_bytes": int(mem["MemTotal"].split()[0]) * 1024,
        "memory_available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "network": net,
        "gpus": gpus,
    }


def actor_class():
    import ray

    @ray.remote
    class Sampler:
        def __init__(self, run_dir: str, interval: float):
            self.stop_event = threading.Event()
            self.path = Path(run_dir) / "metrics" / "nodes" / f"{socket.gethostname()}.jsonl"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self.run, args=(run_dir, interval), daemon=True)
            self.thread.start()

        def run(self, run_dir: str, interval: float) -> None:
            while not self.stop_event.is_set():
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample(run_dir), sort_keys=True) + "\n")
                self.stop_event.wait(interval)

        def stop(self) -> str:
            self.stop_event.set()
            self.thread.join(timeout=15)
            return str(self.path)

    return Sampler


def state_path(run_dir: Path) -> Path:
    return run_dir / "metrics" / "node_sampler_state.json"


def start(args: argparse.Namespace) -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address="auto")
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    if len(nodes) != args.expected_nodes:
        raise SystemExit(f"expected {args.expected_nodes} alive Ray nodes, found {len(nodes)}")
    Sampler = actor_class()
    state = {"schema_version": 1, "actors": []}
    for node in nodes:
        node_id = node["NodeID"]
        name = f"lexbrowser-metrics-{args.run_dir.name}-{node_id[:12]}"
        Sampler.options(
            name=name,
            lifetime="detached",
            num_cpus=0.1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(str(args.run_dir), args.interval)
        state["actors"].append({"node_id": node_id, "name": name})
    state_path(args.run_dir).parent.mkdir(parents=True, exist_ok=True)
    state_path(args.run_dir).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stop(args: argparse.Namespace) -> None:
    import ray

    path = state_path(args.run_dir)
    if not path.exists():
        return
    ray.init(address="auto")
    state = json.loads(path.read_text(encoding="utf-8"))
    stopped = []
    for item in state.get("actors", []):
        try:
            actor = ray.get_actor(item["name"])
            stopped.append(ray.get(actor.stop.remote()))
            ray.kill(actor, no_restart=True)
        except (ValueError, ray.exceptions.RayActorError):
            continue
    (args.run_dir / "metrics" / "node_sampler_stopped.json").write_text(
        json.dumps({"paths": stopped}, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "start":
            command.add_argument("--expected-nodes", type=int, required=True)
            command.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    {"start": start, "stop": stop}[args.command](args)


if __name__ == "__main__":
    main()
