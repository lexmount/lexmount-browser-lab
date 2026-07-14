#!/usr/bin/env python3
"""Aggregate observed resource samples and parser-derived training signals."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def metrics(rows: list[dict]) -> dict[str, float | int | None]:
    rows.sort(key=lambda item: float(item["timestamp_unix"]))
    cpu, ram, disk, network, gpu_util, gpu_mem = [], [], [], [], [], []
    for row in rows:
        total = sum(row.get("cpu_ticks", []))
        idle = sum(row.get("cpu_ticks", [])[3:5])
        row["_cpu_total"], row["_cpu_idle"] = total, idle
        ram.append(1 - float(row["memory_available_bytes"]) / float(row["memory_total_bytes"]))
        disk.append(1 - float(row["disk_free_bytes"]) / float(row["disk_total_bytes"]))
        gpu_util.extend(float(gpu["utilization_pct"]) for gpu in row.get("gpus", []))
        gpu_mem.extend(
            float(gpu["memory_used_mib"]) / float(gpu["memory_total_mib"])
            for gpu in row.get("gpus", [])
            if gpu.get("memory_total_mib")
        )
    for previous, current in zip(rows, rows[1:], strict=False):
        elapsed = float(current["timestamp_unix"]) - float(previous["timestamp_unix"])
        if elapsed <= 0:
            continue
        total_delta = current["_cpu_total"] - previous["_cpu_total"]
        idle_delta = current["_cpu_idle"] - previous["_cpu_idle"]
        if total_delta > 0:
            cpu.append(1 - idle_delta / total_delta)
        for name, values in current.get("network", {}).items():
            earlier = previous.get("network", {}).get(name, {})
            network.append(
                (
                    float(values.get("rx_bytes", 0))
                    - float(earlier.get("rx_bytes", 0))
                    + float(values.get("tx_bytes", 0))
                    - float(earlier.get("tx_bytes", 0))
                )
                / elapsed
            )
    return {
        "samples": len(rows),
        "cpu_utilization_mean": mean(cpu),
        "cpu_utilization_p95": quantile(cpu, 0.95),
        "ram_utilization_mean": mean(ram),
        "ram_utilization_p95": quantile(ram, 0.95),
        "disk_utilization_mean": mean(disk),
        "network_bytes_per_second_mean": mean(network),
        "gpu_utilization_mean": mean(gpu_util),
        "gpu_utilization_p95": quantile(gpu_util, 0.95),
        "gpu_memory_utilization_mean": mean(gpu_mem),
        "gpu_memory_utilization_p95": quantile(gpu_mem, 0.95),
    }


def training_signals(log_root: Path) -> dict[str, float | None]:
    reward, loss = [], []
    for path in log_root.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        reward.extend(float(value) for value in re.findall(r"Avg Reward:\s*([-+0-9.eE]+)", text))
        loss.extend(float(value) for value in re.findall(r"Loss:\s*([-+0-9.eE]+)", text))
    return {
        "last_avg_reward": reward[-1] if reward else None,
        "last_loss": loss[-1] if loss else None,
        "observed_reward_points": len(reward),
        "observed_loss_points": len(loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    by_host: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((args.run_dir / "metrics" / "nodes").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_host[str(row.get("hostname", path.stem))].append(row)
    per_host = {host: metrics(rows) for host, rows in sorted(by_host.items())}
    summary = {
        "schema_version": 1,
        "observed_nodes": len(per_host),
        "per_host": per_host,
        "training": training_signals(args.run_dir / "logs"),
    }
    output = args.run_dir / "metrics" / "resources_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
