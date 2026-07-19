#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, observed)
        * probability**observed
        * (1 - probability) ** (trials - observed)
        for observed in range(successes + 1)
    )


def _one_sided_clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """Return the exact one-sided upper confidence bound for a binomial rate."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial count")
    if successes == trials:
        return 1.0
    if successes == 0:
        return 1 - alpha ** (1 / trials)

    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if _binomial_cdf(successes, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _is_judged(task: dict[str, Any]) -> bool:
    """Treat legacy summaries without a coverage field as fully judged."""
    return bool(task.get("judged", True))


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
    shared_task_ids = sorted(set(lex_tasks) & set(local_tasks))
    task_ids: list[str] = []
    excluded_task_ids = {
        "lexmount_unjudged": [],
        "local_unjudged": [],
        "both_unjudged": [],
    }
    for task_id in shared_task_ids:
        lexmount_judged = _is_judged(lex_tasks[task_id])
        local_judged = _is_judged(local_tasks[task_id])
        if lexmount_judged and local_judged:
            task_ids.append(task_id)
        elif not lexmount_judged and not local_judged:
            excluded_task_ids["both_unjudged"].append(task_id)
        elif not lexmount_judged:
            excluded_task_ids["lexmount_unjudged"].append(task_id)
        else:
            excluded_task_ids["local_unjudged"].append(task_id)
    pairs = [
        (int(bool(lex_tasks[task_id]["success"])), int(bool(local_tasks[task_id]["success"])))
        for task_id in task_ids
    ]
    if not pairs:
        raise ValueError("summaries have no mutually judged task ids")

    lex_success = sum(pair[0] for pair in pairs)
    local_success = sum(pair[1] for pair in pairs)
    both_success = sum(pair == (1, 1) for pair in pairs)
    lex_only = sum(pair == (1, 0) for pair in pairs)
    local_only = sum(pair == (0, 1) for pair in pairs)
    both_failed = len(pairs) - both_success - lex_only - local_only
    difference = (lex_success - local_success) / len(pairs)
    local_only_upper = _one_sided_clopper_pearson_upper(local_only, len(pairs))

    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        bootstrap.append(sum(left - right for left, right in sample) / len(sample))
    lower = _quantile(bootstrap, 0.025)
    upper = _quantile(bootstrap, 0.975)
    return {
        "schema_version": 2,
        "paired_tasks": len(pairs),
        "coverage": {
            "shared_planned_tasks": len(shared_task_ids),
            "mutually_judged_tasks": len(task_ids),
            "excluded_task_ids": excluded_task_ids,
        },
        "success": {"lexmount": lex_success, "local": local_success},
        "paired_table": {
            "both_success": both_success,
            "lexmount_only": lex_only,
            "local_only": local_only,
            "both_failed": both_failed,
        },
        "positive_outcome_coverage": {
            "both_success": both_success,
            "at_least_one_success": both_success + lex_only + local_only,
            "both_failed": both_failed,
        },
        "success_rate_difference": {
            "lexmount_minus_local": round(difference, 6),
            "bootstrap_95_ci": [round(lower, 6), round(upper, 6)],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_note": "descriptive resampling interval; not used for noninferiority",
        },
        "noninferiority": {
            "margin": margin,
            "local_only_count": local_only,
            "local_only_rate": round(local_only / len(pairs), 6),
            "local_only_one_sided_95_upper_bound": round(local_only_upper, 6),
            "passed": local_only_upper <= margin,
            "rule": "exact one-sided Clopper-Pearson upper bound for local-only outcomes <= margin",
            "scope_note": (
                "A pass bounds the rate of observed Local-only successes; "
                "positive shared-success coverage is reported separately."
            ),
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
