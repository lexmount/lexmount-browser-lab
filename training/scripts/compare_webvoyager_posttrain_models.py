#!/usr/bin/env python3
"""Compare base and trained WebVoyager checkpoints under a fixed environment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from audit_webvoyager_posttrain_pair import (
    CONTRACT_KEYS,
    arm_metrics,
    index_results,
    load_json,
    normalize_arm,
    resource_summary,
)

CONTROL_KEYS = tuple(key for key in CONTRACT_KEYS if key != "model")


def controls_match(
    base_manifest: Mapping[str, Any], trained_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    differences = {
        key: {"base": base_manifest.get(key), "trained": trained_manifest.get(key)}
        for key in CONTROL_KEYS
        if base_manifest.get(key) != trained_manifest.get(key)
    }
    return {"matches": not differences, "differences": differences}


def model_outcome(base: str | None, trained: str | None) -> str:
    if base not in {"yes", "no"} or trained not in {"yes", "no"}:
        return "unjudged"
    if base == "yes" and trained == "yes":
        return "both_success"
    if trained == "yes":
        return "trained_only_success"
    if base == "yes":
        return "base_only_success"
    return "both_no"


def compare_checkpoints(base_dir: Path, trained_dir: Path) -> dict[str, Any]:
    base_manifest = load_json(base_dir / "run_manifest.json")
    trained_manifest = load_json(trained_dir / "run_manifest.json")
    base_records = index_results(base_dir / "results.jsonl")
    trained_records = index_results(trained_dir / "results.jsonl")
    base_ids = set(base_records)
    trained_ids = set(trained_records)
    if base_ids != trained_ids:
        raise ValueError(
            "checkpoint task coverage differs: "
            f"base_only={sorted(base_ids - trained_ids)}; "
            f"trained_only={sorted(trained_ids - base_ids)}"
        )

    pairs: list[dict[str, Any]] = []
    for identifier in sorted(base_ids):
        base_arm = normalize_arm(base_records[identifier])
        trained_arm = normalize_arm(trained_records[identifier])
        if (base_arm["website"], base_arm["split"]) != (
            trained_arm["website"],
            trained_arm["split"],
        ):
            raise ValueError(f"task metadata differs between checkpoints for {identifier}")
        raw_outcome = model_outcome(base_arm["judge_verdict"], trained_arm["judge_verdict"])
        pair_eligible = bool(base_arm["quality_eligible"] and trained_arm["quality_eligible"])
        pairs.append(
            {
                "task_id": identifier,
                "website": base_arm["website"],
                "split": base_arm["split"],
                "quality_pair_eligible": pair_eligible,
                "raw_judge_outcome": raw_outcome,
                "quality_judge_outcome": raw_outcome if pair_eligible else "ineligible",
                "base": base_arm,
                "trained": trained_arm,
            }
        )

    base_arms = [pair["base"] for pair in pairs]
    trained_arms = [pair["trained"] for pair in pairs]
    quality_pairs = [pair for pair in pairs if pair["quality_pair_eligible"]]
    base_successes = sum(pair["base"]["judge_verdict"] == "yes" for pair in quality_pairs)
    trained_successes = sum(pair["trained"]["judge_verdict"] == "yes" for pair in quality_pairs)
    eligible = len(quality_pairs)
    return {
        "schema_version": 1,
        "control_contract": controls_match(base_manifest, trained_manifest),
        "models": {"base": base_manifest.get("model"), "trained": trained_manifest.get("model")},
        "tasks": len(pairs),
        "arms": {"base": arm_metrics(base_arms), "trained": arm_metrics(trained_arms)},
        "paired_quality": {
            "eligible_tasks": eligible,
            "outcomes": dict(
                sorted(Counter(pair["quality_judge_outcome"] for pair in quality_pairs).items())
            ),
            "base_successes": base_successes,
            "trained_successes": trained_successes,
            "trained_minus_base_success_rate": (
                (trained_successes - base_successes) / eligible if eligible else None
            ),
        },
        "raw_judge_outcomes": dict(
            sorted(Counter(pair["raw_judge_outcome"] for pair in pairs).items())
        ),
        "resources": {
            "base": resource_summary(base_dir),
            "trained": resource_summary(trained_dir),
        },
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare base and trained WebVoyager checkpoint evaluations."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--trained-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict-controls",
        action="store_true",
        help="fail instead of writing a comparison when non-model controls differ",
    )
    args = parser.parse_args()
    comparison = compare_checkpoints(args.base_dir, args.trained_dir)
    if args.strict_controls and not comparison["control_contract"]["matches"]:
        raise SystemExit("checkpoint controls differ; refusing a strict comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
