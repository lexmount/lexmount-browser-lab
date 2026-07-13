#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from .compare_pair import compare_pair


def _success_map(summary: dict[str, Any], name: str) -> dict[str, int]:
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict) or not per_task:
        raise ValueError(f"{name}: missing or empty per_task")
    output: dict[str, int] = {}
    for task_id, record in per_task.items():
        if not isinstance(record, dict) or "success" not in record:
            raise ValueError(f"{name}: task {task_id} is missing success")
        output[str(task_id)] = int(bool(record["success"]))
    return output


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _repeatability(runs: list[dict[str, int]], task_ids: list[str]) -> dict[str, Any]:
    sequences = {task_id: [run[task_id] for run in runs] for task_id in task_ids}
    all_success = sum(all(values) for values in sequences.values())
    all_failed = sum(not any(values) for values in sequences.values())
    mixed = len(task_ids) - all_success - all_failed
    return {
        "replicates": len(runs),
        "all_success": all_success,
        "all_failed": all_failed,
        "mixed": mixed,
        "unanimous_rate": round((all_success + all_failed) / len(task_ids), 6),
    }


def compare_repeated_pairs(
    lexmount_summaries: list[dict[str, Any]],
    local_summaries: list[dict[str, Any]],
    *,
    labels: list[str] | None = None,
    bootstrap_samples: int = 100_000,
    seed: int = 55,
) -> dict[str, Any]:
    if not lexmount_summaries or len(lexmount_summaries) != len(local_summaries):
        raise ValueError("Lexmount and Local must have the same non-zero replicate count")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if labels is None:
        labels = [f"replicate-{index + 1}" for index in range(len(lexmount_summaries))]
    if len(labels) != len(lexmount_summaries) or len(set(labels)) != len(labels):
        raise ValueError("labels must be unique and match the replicate count")

    lexmount_runs = [
        _success_map(summary, f"Lexmount {labels[index]}")
        for index, summary in enumerate(lexmount_summaries)
    ]
    local_runs = [
        _success_map(summary, f"Local {labels[index]}")
        for index, summary in enumerate(local_summaries)
    ]
    task_ids = sorted(lexmount_runs[0], key=int)
    expected = set(task_ids)
    for name, run in [
        *[(f"Lexmount {labels[index]}", run) for index, run in enumerate(lexmount_runs)],
        *[(f"Local {labels[index]}", run) for index, run in enumerate(local_runs)],
    ]:
        if set(run) != expected:
            raise ValueError(f"{name}: task coverage differs from the first replicate")

    per_replicate: dict[str, Any] = {}
    replicate_differences: list[float] = []
    for index, label in enumerate(labels):
        comparison = compare_pair(
            lexmount_summaries[index],
            local_summaries[index],
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        per_replicate[label] = comparison
        replicate_differences.append(
            float(comparison["success_rate_difference"]["lexmount_minus_local"])
        )

    task_cluster_differences = [
        statistics.fmean(
            lexmount_runs[index][task_id] - local_runs[index][task_id]
            for index in range(len(labels))
        )
        for task_id in task_ids
    ]
    difference = statistics.fmean(task_cluster_differences)
    rng = random.Random(seed)
    bootstrap = [
        statistics.fmean(
            task_cluster_differences[rng.randrange(len(task_cluster_differences))]
            for _ in task_cluster_differences
        )
        for _ in range(bootstrap_samples)
    ]
    lexmount_successes = sum(sum(run.values()) for run in lexmount_runs)
    local_successes = sum(sum(run.values()) for run in local_runs)
    attempts = len(task_ids) * len(labels)
    task_records = [
        {
            "task_id": task_id,
            "lexmount": [run[task_id] for run in lexmount_runs],
            "local": [run[task_id] for run in local_runs],
            "mean_lexmount_minus_local": round(task_cluster_differences[index], 6),
        }
        for index, task_id in enumerate(task_ids)
    ]
    return {
        "schema_version": 1,
        "paired_tasks": len(task_ids),
        "replicates": len(labels),
        "paired_attempts": attempts,
        "labels": labels,
        "success_attempts": {
            "lexmount": lexmount_successes,
            "local": local_successes,
        },
        "success_rates": {
            "lexmount": round(lexmount_successes / attempts, 6),
            "local": round(local_successes / attempts, 6),
        },
        "clustered_difference": {
            "lexmount_minus_local": round(difference, 6),
            "task_cluster_bootstrap_95_ci": [
                round(_quantile(bootstrap, 0.025), 6),
                round(_quantile(bootstrap, 0.975), 6),
            ],
            "bootstrap_samples": bootstrap_samples,
            "cluster_unit": "task_id",
        },
        "per_replicate": per_replicate,
        "replicate_difference_range": {
            "min": round(min(replicate_differences), 6),
            "max": round(max(replicate_differences), 6),
            "span": round(max(replicate_differences) - min(replicate_differences), 6),
        },
        "repeatability": {
            "lexmount": _repeatability(lexmount_runs, task_ids),
            "local": _repeatability(local_runs, task_ids),
        },
        "per_task": task_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare repeated paired browser runs")
    parser.add_argument("--lexmount", type=Path, action="append", required=True)
    parser.add_argument("--local", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lexmount = [json.loads(path.read_text(encoding="utf-8")) for path in args.lexmount]
    local = [json.loads(path.read_text(encoding="utf-8")) for path in args.local]
    result = compare_repeated_pairs(
        lexmount,
        local,
        labels=args.label,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
