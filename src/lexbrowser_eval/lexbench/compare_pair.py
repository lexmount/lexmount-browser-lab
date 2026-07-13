#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def compare_pair(
    lexmount: dict[str, Any],
    local: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 55,
    margin: float = 0.05,
) -> dict[str, Any]:
    lex_tasks = lexmount["per_task"]
    local_tasks = local["per_task"]
    task_ids = sorted(set(lex_tasks) & set(local_tasks))
    pairs = [
        (int(bool(lex_tasks[task_id]["success"])), int(bool(local_tasks[task_id]["success"])))
        for task_id in task_ids
    ]
    if not pairs:
        raise ValueError("summaries have no shared task ids")

    lex_success = sum(pair[0] for pair in pairs)
    local_success = sum(pair[1] for pair in pairs)
    both_success = sum(pair == (1, 1) for pair in pairs)
    lex_only = sum(pair == (1, 0) for pair in pairs)
    local_only = sum(pair == (0, 1) for pair in pairs)
    difference = (lex_success - local_success) / len(pairs)

    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        bootstrap.append(sum(left - right for left, right in sample) / len(sample))
    lower = _quantile(bootstrap, 0.025)
    upper = _quantile(bootstrap, 0.975)
    return {
        "schema_version": 1,
        "paired_tasks": len(pairs),
        "success": {"lexmount": lex_success, "local": local_success},
        "paired_table": {
            "both_success": both_success,
            "lexmount_only": lex_only,
            "local_only": local_only,
            "both_failed": len(pairs) - both_success - lex_only - local_only,
        },
        "success_rate_difference": {
            "lexmount_minus_local": round(difference, 6),
            "bootstrap_95_ci": [round(lower, 6), round(upper, 6)],
            "bootstrap_samples": bootstrap_samples,
        },
        "noninferiority": {
            "margin": margin,
            "passed": lower >= -margin,
            "rule": "lower bootstrap CI >= -margin",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare paired Lexmount and local summaries")
    parser.add_argument("--lexmount", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    lexmount = json.loads(args.lexmount.read_text(encoding="utf-8"))
    local = json.loads(args.local.read_text(encoding="utf-8"))
    comparison = compare_pair(
        lexmount,
        local,
        bootstrap_samples=args.bootstrap_samples,
        margin=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
