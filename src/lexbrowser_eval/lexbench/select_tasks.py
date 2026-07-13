#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _task_id(task: dict[str, Any]) -> str:
    value = task.get("id") or task.get("task_id")
    if value in (None, ""):
        raise ValueError("task is missing id/task_id")
    return str(value)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            _task_id(value)
            tasks.append(value)
    return tasks


def _allocate_strata(sizes: dict[tuple[str, str], int], count: int) -> dict[tuple[str, str], int]:
    total = sum(sizes.values())
    if count < 1 or count > total:
        raise ValueError(f"count must be between 1 and {total}")

    exact = {key: count * size / total for key, size in sizes.items()}
    allocated = {key: min(size, math.floor(exact[key])) for key, size in sizes.items()}
    remaining = count - sum(allocated.values())
    order = sorted(
        sizes,
        key=lambda key: (exact[key] - allocated[key], sizes[key], key),
        reverse=True,
    )
    for key in order:
        if remaining == 0:
            break
        if allocated[key] < sizes[key]:
            allocated[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"failed to allocate {remaining} tasks")
    return allocated


def select_tasks(tasks: list[dict[str, Any]], count: int, seed: str) -> list[str]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        key = (
            str(task.get("website_region") or "unknown"),
            str(task.get("task_type") or "unknown"),
        )
        strata[key].append(task)

    allocation = _allocate_strata({key: len(values) for key, values in strata.items()}, count)
    selected: list[str] = []
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda task: hashlib.sha256(f"{seed}:{_task_id(task)}".encode()).hexdigest(),
        )
        selected.extend(_task_id(task) for task in ranked[: allocation[key]])
    return sorted(
        selected,
        key=lambda task_id: hashlib.sha256(f"{seed}:order:{task_id}".encode()).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a deterministic region/type-stratified task set"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", default="gpt55-lexbench-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = select_tasks(load_tasks(args.dataset), args.count, args.seed)
    payload = "\n".join(selected) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
