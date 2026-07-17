#!/usr/bin/env python3
"""Aggregate repeated paired WebVoyager browser-environment evaluations.

Each input directory contains one Lexmount and one local-Chrome run produced
under the same checkpoint and task manifest. Repetitions may intentionally
use different policy sampling seeds; every other control must remain fixed.
The result reports repeated observations separately from the number of
distinct benchmark tasks, so retries cannot be mistaken for a larger suite.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audit_webvoyager_posttrain_pair import arm_metrics, audit_pair, load_json

CONTROL_KEYS = (
    "protocol",
    "schema_version",
    "tasks",
    "tasks_sha256",
    "selected_tasks",
    "evaluator",
    "judge",
    "model",
    "browser",
)


def run_control_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return controls that must match between repeats, excluding seed only."""

    generation = manifest.get("generation")
    normalized_generation = (
        {key: value for key, value in generation.items() if key != "seed_base"}
        if isinstance(generation, Mapping)
        else generation
    )
    return {
        **{key: manifest.get(key) for key in CONTROL_KEYS},
        "generation": normalized_generation,
    }


def control_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in reference.keys() | candidate.keys()
        if reference.get(key) != candidate.get(key)
    }


def task_signature(pairs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    return sorted(
        (
            str(pair["task_id"]),
            str(pair.get("website") or ""),
            str(pair.get("split") or ""),
        )
        for pair in pairs
    )


def outcome_rate(successes: int, observations: int) -> float | None:
    return successes / observations if observations else None


def aggregate_repeats(run_dirs: Sequence[Path]) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one --run-dir is required")
    resolved_dirs = [path.resolve() for path in run_dirs]
    if len(set(resolved_dirs)) != len(resolved_dirs):
        raise ValueError("duplicate --run-dir values are not allowed")

    audits: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for run_dir in resolved_dirs:
        lexmount_dir = run_dir / "lexmount"
        local_dir = run_dir / "local"
        audit = audit_pair(lexmount_dir, local_dir)
        manifest = load_json(lexmount_dir / "run_manifest.json")
        audits.append((run_dir, audit, manifest))

    reference_tasks = task_signature(audits[0][1]["pairs"])
    reference_control = run_control_manifest(audits[0][2])
    control_mismatches: dict[str, dict[str, Any]] = {}
    pair_contract_mismatches: dict[str, Any] = {}
    for run_dir, audit, manifest in audits:
        signature = task_signature(audit["pairs"])
        if signature != reference_tasks:
            raise ValueError(
                "repeated task coverage differs: "
                f"reference={reference_tasks}; {run_dir}={signature}"
            )
        differences = control_differences(reference_control, run_control_manifest(manifest))
        if differences:
            control_mismatches[str(run_dir)] = differences
        pair_contract = audit["comparison_contract"]
        if not pair_contract["matches"]:
            pair_contract_mismatches[str(run_dir)] = pair_contract["differences"]

    all_pairs = [pair for _, audit, _ in audits for pair in audit["pairs"]]
    lexmount_arms = [pair["lexmount"] for pair in all_pairs]
    local_arms = [pair["local"] for pair in all_pairs]
    quality_pairs = [pair for pair in all_pairs if pair["quality_pair_eligible"]]
    quality_outcomes = Counter(pair["quality_judge_outcome"] for pair in quality_pairs)
    lexmount_successes = sum(
        pair["lexmount"]["judge_verdict"] == "yes" for pair in quality_pairs
    )
    local_successes = sum(
        pair["local"]["judge_verdict"] == "yes" for pair in quality_pairs
    )

    per_task_pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in all_pairs:
        per_task_pairs[str(pair["task_id"])].append(pair)
    per_task = []
    for task_id in sorted(per_task_pairs):
        pairs = per_task_pairs[task_id]
        eligible = [pair for pair in pairs if pair["quality_pair_eligible"]]
        per_task.append(
            {
                "task_id": task_id,
                "website": pairs[0]["website"],
                "split": pairs[0]["split"],
                "observations": len(pairs),
                "quality_pair_eligible_observations": len(eligible),
                "raw_judge_outcomes": dict(
                    sorted(Counter(pair["raw_judge_outcome"] for pair in pairs).items())
                ),
                "quality_judge_outcomes": dict(
                    sorted(Counter(pair["quality_judge_outcome"] for pair in eligible).items())
                ),
                "lexmount_quality_successes": sum(
                    pair["lexmount"]["judge_verdict"] == "yes" for pair in eligible
                ),
                "local_quality_successes": sum(
                    pair["local"]["judge_verdict"] == "yes" for pair in eligible
                ),
            }
        )

    runs = []
    for run_dir, audit, manifest in audits:
        generation = manifest.get("generation")
        seed_base = generation.get("seed_base") if isinstance(generation, Mapping) else None
        runs.append(
            {
                "run_dir": str(run_dir),
                "seed_base": seed_base,
                "pair_contract": audit["comparison_contract"],
                "arms": audit["arms"],
                "paired_quality": audit["paired_quality"],
                "resources": audit["resources"],
            }
        )

    eligible_observations = len(quality_pairs)
    return {
        "schema_version": 1,
        "repeat_control_contract": {
            "matches": not control_mismatches and not pair_contract_mismatches,
            "allowed_variation": ["generation.seed_base"],
            "control_differences": control_mismatches,
            "within_pair_differences": pair_contract_mismatches,
        },
        "distinct_tasks": len(reference_tasks),
        "repeat_runs": len(audits),
        "paired_observations": len(all_pairs),
        "arms": {
            "lexmount": arm_metrics(lexmount_arms),
            "local": arm_metrics(local_arms),
        },
        "paired_quality": {
            "eligible_observations": eligible_observations,
            "outcomes": dict(sorted(quality_outcomes.items())),
            "lexmount_successes": lexmount_successes,
            "local_successes": local_successes,
            "lexmount_success_rate": outcome_rate(lexmount_successes, eligible_observations),
            "local_success_rate": outcome_rate(local_successes, eligible_observations),
            "lexmount_minus_local_success_rate": outcome_rate(
                lexmount_successes - local_successes, eligible_observations
            ),
        },
        "raw_judge_outcomes": dict(
            sorted(Counter(pair["raw_judge_outcome"] for pair in all_pairs).items())
        ),
        "per_task": per_task,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated matched WebVoyager Lexmount/local evaluations."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="directory containing lexmount/ and local/ run directories; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict-contract",
        action="store_true",
        help="refuse output if any within-run or cross-repeat controls differ",
    )
    args = parser.parse_args()
    aggregate = aggregate_repeats(args.run_dir)
    if args.strict_contract and not aggregate["repeat_control_contract"]["matches"]:
        raise SystemExit("repeat controls differ; refusing a strict aggregation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
