#!/usr/bin/env python3
"""Create a reproducible, stratified LexBench task subset.

The generated block A and block B files contain disjoint halves of the same
sample.  They are intended to be run with opposite browser-backend orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--prefix", default="stratified64")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocate(counts: Counter[tuple[str, str]], size: int) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    if not 0 < size <= total:
        raise ValueError(f"size must be in [1, {total}], got {size}")

    allocation: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    for stratum, count in sorted(counts.items()):
        exact = size * count / total
        allocation[stratum] = int(exact)
        remainders.append((exact - allocation[stratum], stratum))

    for _, stratum in sorted(remainders, reverse=True)[: size - sum(allocation.values())]:
        allocation[stratum] += 1
    return allocation


def write_task_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(str(record["id"]) for record in records) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("dataset contains duplicate task ids")

    strata = Counter((record["language"], record["task_type"]) for record in records)
    allocation = allocate(strata, args.size)
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    block_a: list[dict[str, Any]] = []
    block_b: list[dict[str, Any]] = []

    for stratum in sorted(allocation):
        pool = [record for record in records if (record["language"], record["task_type"]) == stratum]
        rng.shuffle(pool)
        picked = pool[: allocation[stratum]]
        selected.extend(picked)
        block_a.extend(picked[::2])
        block_b.extend(picked[1::2])

    rng.shuffle(block_a)
    rng.shuffle(block_b)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / f"{args.prefix}-all.txt"
    a_path = args.output_dir / f"{args.prefix}-block-a.txt"
    b_path = args.output_dir / f"{args.prefix}-block-b.txt"
    write_task_file(all_path, selected)
    write_task_file(a_path, block_a)
    write_task_file(b_path, block_b)

    manifest = {
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "seed": args.seed,
        "sample_size": args.size,
        "strata": {f"{language}/{task_type}": count for (language, task_type), count in sorted(strata.items())},
        "allocation": {f"{language}/{task_type}": count for (language, task_type), count in sorted(allocation.items())},
        "blocks": {
            "a": {"count": len(block_a), "sha256": sha256(a_path)},
            "b": {"count": len(block_b), "sha256": sha256(b_path)},
        },
        "sample_sha256": sha256(all_path),
        "selection": "largest-remainder proportional allocation by language/task_type, then seeded shuffle within each stratum",
    }
    manifest_path = args.output_dir / f"{args.prefix}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
