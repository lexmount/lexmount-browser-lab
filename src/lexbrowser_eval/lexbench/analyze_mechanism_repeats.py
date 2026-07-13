#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

OUTCOME_CODES = {
    (True, True): "B",
    (True, False): "X",
    (False, True): "L",
    (False, False): "F",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _success_map(summary: dict[str, Any], name: str) -> dict[str, bool]:
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict) or not per_task:
        raise ValueError(f"{name}: missing or empty per_task")
    output: dict[str, bool] = {}
    for task_id, record in per_task.items():
        if not isinstance(record, dict) or "success" not in record:
            raise ValueError(f"{name}: task {task_id} is missing success")
        output[str(task_id)] = bool(record["success"])
    return output


def _selection_map(selection: dict[str, Any]) -> dict[str, dict[str, str]]:
    groups = selection.get("selection")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("selection: missing selection groups")
    output: dict[str, dict[str, str]] = {}
    for category, records in groups.items():
        if not isinstance(records, list):
            raise ValueError(f"selection: category {category} must be a list")
        for record in records:
            if not isinstance(record, dict) or not record.get("task_id"):
                raise ValueError(f"selection: category {category} has an invalid record")
            task_id = str(record["task_id"])
            if task_id in output:
                raise ValueError(f"selection: duplicate task {task_id}")
            output[task_id] = {
                "category": str(category),
                "initial_pattern": str(record.get("pair_outcomes") or ""),
                "language": str(record.get("language") or "unknown"),
            }
    return output


def _repeatability(runs: list[dict[str, bool]], task_ids: list[str]) -> dict[str, Any]:
    all_success = sum(all(run[task_id] for run in runs) for task_id in task_ids)
    all_failed = sum(not any(run[task_id] for run in runs) for task_id in task_ids)
    mixed = len(task_ids) - all_success - all_failed
    return {
        "all_success": all_success,
        "all_failed": all_failed,
        "mixed": mixed,
        "unanimous_rate": round((all_success + all_failed) / len(task_ids), 6),
    }


def _audit_observations(
    audits: list[dict[str, Any]], labels: list[str], task_ids: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    bucket_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    error_signature_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    raw_indicator_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    outcome_counts: Counter[str] = Counter()
    near_threshold = 0
    high_similarity = 0
    for label, audit in zip(labels, audits, strict=True):
        discordant = audit.get("discordant")
        if not isinstance(discordant, list):
            raise ValueError(f"audit {label}: missing discordant list")
        for record in discordant:
            if not isinstance(record, dict) or not record.get("task_id"):
                raise ValueError(f"audit {label}: invalid discordant record")
            task_id = str(record["task_id"])
            if task_id not in task_ids:
                continue
            outcome = str(record.get("outcome") or "unknown")
            bucket = str(record.get("evidence_bucket") or "unresolved")
            outcome_counts[outcome] += 1
            loser = (
                record.get("local") or {}
                if outcome == "lexmount_only"
                else record.get("lexmount") or {}
            )
            if outcome in bucket_counts:
                bucket_counts[outcome][bucket] += 1
                error_signature_counts[outcome][
                    str(loser.get("error_signature") or "none")
                ] += 1
                raw_indicator_counts[outcome].update(loser.get("raw_log_indicators") or [])
            near_threshold += int(bool(record.get("near_threshold_loser")))
            high_similarity += int(bool(record.get("high_answer_similarity")))
            by_task[task_id].append(
                {
                    "label": label,
                    "outcome": outcome,
                    "target_website": record.get("target_website"),
                    "evidence_bucket": bucket,
                    "near_threshold_loser": bool(record.get("near_threshold_loser")),
                    "high_answer_similarity": bool(record.get("high_answer_similarity")),
                    "loser_failure_category": loser.get("failure_category"),
                    "loser_error_signature": loser.get("error_signature"),
                    "loser_raw_log_indicators": loser.get("raw_log_indicators") or [],
                }
            )
    return by_task, {
        "discordant_observations": sum(outcome_counts.values()),
        "outcomes": dict(outcome_counts),
        "loser_evidence_buckets": {
            outcome: dict(counts) for outcome, counts in bucket_counts.items()
        },
        "loser_error_signatures": {
            outcome: dict(counts) for outcome, counts in error_signature_counts.items()
        },
        "loser_raw_log_indicators": {
            outcome: dict(counts) for outcome, counts in raw_indicator_counts.items()
        },
        "near_threshold_loser": near_threshold,
        "high_answer_similarity": high_similarity,
    }


def analyze_mechanism_repeats(
    selection: dict[str, Any],
    lexmount_summaries: list[dict[str, Any]],
    local_summaries: list[dict[str, Any]],
    *,
    labels: list[str],
    audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(labels) < 3:
        raise ValueError("mechanism analysis requires at least three repeated pairs")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    if len(lexmount_summaries) != len(labels) or len(local_summaries) != len(labels):
        raise ValueError("summary counts must match labels")
    if audits is not None and len(audits) != len(labels):
        raise ValueError("audit count must match labels")

    selected = _selection_map(selection)
    task_ids = sorted(selected, key=int)
    if not task_ids:
        raise ValueError("selection contains no task records")
    expected = set(task_ids)
    lexmount_runs = [
        _success_map(summary, f"Lexmount {label}")
        for label, summary in zip(labels, lexmount_summaries, strict=True)
    ]
    local_runs = [
        _success_map(summary, f"Local {label}")
        for label, summary in zip(labels, local_summaries, strict=True)
    ]
    for name, run in [
        *[(f"Lexmount {label}", run) for label, run in zip(labels, lexmount_runs, strict=True)],
        *[(f"Local {label}", run) for label, run in zip(labels, local_runs, strict=True)],
    ]:
        missing = expected - set(run)
        if missing:
            raise ValueError(
                f"{name}: missing selected tasks {','.join(sorted(missing, key=int))}"
            )

    audit_by_task: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    evidence: dict[str, Any] | None = None
    if audits is not None:
        audit_by_task, evidence = _audit_observations(audits, labels, expected)

    per_replicate: dict[str, Any] = {}
    patterns: dict[str, str] = {}
    per_task: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        counts: Counter[str] = Counter()
        for task_id in task_ids:
            counts[OUTCOME_CODES[(lexmount_runs[index][task_id], local_runs[index][task_id])]] += 1
        per_replicate[label] = {
            "outcomes": dict(counts),
            "lexmount_success": sum(lexmount_runs[index][task_id] for task_id in task_ids),
            "local_success": sum(local_runs[index][task_id] for task_id in task_ids),
        }

    for task_id in task_ids:
        pattern = "".join(
            OUTCOME_CODES[(lexmount_runs[index][task_id], local_runs[index][task_id])]
            for index in range(len(labels))
        )
        initial_pattern = selected[task_id]["initial_pattern"]
        if initial_pattern and not pattern.startswith(initial_pattern):
            raise ValueError(
                f"task {task_id}: observed prefix {pattern[:len(initial_pattern)]} "
                f"does not match selected pattern {initial_pattern}"
            )
        patterns[task_id] = pattern
        per_task.append(
            {
                "task_id": task_id,
                **selected[task_id],
                "target_website": next(
                    (
                        observation["target_website"]
                        for observation in audit_by_task[task_id]
                        if observation.get("target_website")
                    ),
                    None,
                ),
                "pattern": pattern,
                "followup_pattern": pattern[len(initial_pattern) :],
                "lexmount_successes": sum(run[task_id] for run in lexmount_runs),
                "local_successes": sum(run[task_id] for run in local_runs),
                "audit_observations": audit_by_task[task_id],
            }
        )

    category_summary: dict[str, Any] = {}
    for category in sorted({record["category"] for record in selected.values()}):
        records = [record for record in per_task if record["category"] == category]
        followup_counts = Counter(code for record in records for code in record["followup_pattern"])
        item: dict[str, Any] = {
            "tasks": len(records),
            "followup_observations": sum(followup_counts.values()),
            "followup_outcomes": dict(followup_counts),
            "unanimous_all_repeats": sum(len(set(record["pattern"])) == 1 for record in records),
        }
        expected_code = {
            "stable_lexmount_only": "X",
            "stable_local_only": "L",
        }.get(category)
        if expected_code:
            item.update(
                {
                    "expected_outcome": expected_code,
                    "followup_expected_observations": followup_counts[expected_code],
                    "followup_expected_rate": round(
                        followup_counts[expected_code] / max(1, sum(followup_counts.values())), 6
                    ),
                    "tasks_expected_in_all_repeats": sum(
                        set(record["pattern"]) == {expected_code} for record in records
                    ),
                    "tasks_expected_in_all_followups": sum(
                        bool(record["followup_pattern"])
                        and set(record["followup_pattern"]) == {expected_code}
                        for record in records
                    ),
                }
            )
        category_summary[category] = item

    attempts = len(task_ids) * len(labels)
    lexmount_successes = sum(run[task_id] for run in lexmount_runs for task_id in task_ids)
    local_successes = sum(run[task_id] for run in local_runs for task_id in task_ids)
    return {
        "schema_version": 1,
        "purpose": "diagnostic mechanism repeat analysis; not a population estimate",
        "labels": labels,
        "tasks": len(task_ids),
        "replicates": len(labels),
        "attempts_per_backend": attempts,
        "success_attempts": {
            "lexmount": lexmount_successes,
            "local": local_successes,
        },
        "success_rates": {
            "lexmount": round(lexmount_successes / attempts, 6),
            "local": round(local_successes / attempts, 6),
        },
        "per_replicate": per_replicate,
        "repeatability": {
            "lexmount": _repeatability(lexmount_runs, task_ids),
            "local": _repeatability(local_runs, task_ids),
            "same_pair_outcome_all_repeats": sum(
                len(set(pattern)) == 1 for pattern in patterns.values()
            ),
            "contains_both_lexmount_only_and_local_only": sum(
                "X" in pattern and "L" in pattern for pattern in patterns.values()
            ),
        },
        "category_summary": category_summary,
        "evidence": evidence,
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze repeated diagnostic browser pairs")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--lexmount", type=Path, action="append", required=True)
    parser.add_argument("--local", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_mechanism_repeats(
        _load_object(args.selection),
        [_load_object(path) for path in args.lexmount],
        [_load_object(path) for path in args.local],
        labels=args.label,
        audits=[_load_object(path) for path in args.audit] if args.audit else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
