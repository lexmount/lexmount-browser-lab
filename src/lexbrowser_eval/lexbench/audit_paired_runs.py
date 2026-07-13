#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

INDICATOR_PATTERNS = {
    "captcha_or_bot": re.compile(
        r"captcha|cloudflare|verify (?:you are|that you are) human|human verification|"
        r"unusual traffic|人机验证|验证码|机器人验证",
        re.IGNORECASE,
    ),
    "http_access_denied": re.compile(
        r"HTTP(?: ERROR)?\s*(?:401|403|429)\b|"
        r"\b(?:401|403|429)\s+(?:Unauthorized|Forbidden|Too Many Requests)\b|"
        r"forbidden|access denied|ERR_HTTP_RESPONSE_CODE_FAILURE|访问被拒绝|拒绝访问",
        re.IGNORECASE,
    ),
    "network_navigation": re.compile(
        r"ERR_(?:NETWORK|CONNECTION|NAME_NOT_RESOLVED|TIMED_OUT|TUNNEL|PROXY)|"
        r"navigation failed|net::",
        re.IGNORECASE,
    ),
}


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON record") from exc
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON object, array, or JSON Lines")
    records: list[dict[str, Any]] = []
    for record_number, record in enumerate(value, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {record_number} is not a JSON object")
        records.append(record)
    return records


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def index_records(path: Path, key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record_number, record in enumerate(load_json_records(path), start=1):
        value = record.get(key)
        if value is None or value == "":
            raise ValueError(f"{path}: record {record_number} is missing '{key}'")
        record_id = str(value)
        if record_id in indexed:
            raise ValueError(f"{path}: duplicate '{key}' value {record_id}")
        indexed[record_id] = record
    return indexed


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    return index_records(path, "id")


def load_evaluations(path: Path) -> dict[str, dict[str, Any]]:
    return index_records(path, "task_id")


def load_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        path.parent.name: load_json_object(path)
        for path in sorted((run_dir / "tasks").glob("*/result.json"))
    }


def summary_task_records(summary: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict):
        raise ValueError(f"{name}: missing or invalid 'per_task' object")
    for task_id, task in per_task.items():
        if not isinstance(task, dict):
            raise ValueError(f"{name}: task {task_id} is not a JSON object")
        if "success" not in task:
            raise ValueError(f"{name}: task {task_id} is missing 'success'")
    return per_task


def answer_similarity(left: str | None, right: str | None) -> float | None:
    if not left or not right:
        return None
    normalized_left = re.sub(r"\s+", "", left)[:20_000]
    normalized_right = re.sub(r"\s+", "", right)[:20_000]
    if not normalized_left or not normalized_right:
        return None
    return round(SequenceMatcher(None, normalized_left, normalized_right).ratio(), 4)


def raw_log_indicators(result: dict[str, Any]) -> list[str]:
    text = json.dumps(
        {"error": result.get("error"), "action_history": result.get("action_history")},
        ensure_ascii=False,
    )
    return sorted(name for name, pattern in INDICATOR_PATTERNS.items() if pattern.search(text))


def error_signature(result: dict[str, Any]) -> str | None:
    text = str(result.get("error") or "").strip().lower()
    if not text:
        return None
    if "agent stopped before completion" in text:
        return "agent_stopped_without_done"
    if "expected at least one handler to return" in text or "browserstaterequestevent" in text:
        return "browser_state_handler_failure"
    if "navigation failed" in text:
        return "navigation_failure"
    if "timeout after" in text or "timed out" in text:
        return "timeout"
    if "session" in text and ("create" in text or "connect" in text):
        return "session_lifecycle_failure"
    return "other_error"


def failure_metadata(evaluation: dict[str, Any]) -> tuple[str | None, list[str]]:
    details = evaluation.get("evaluation_details") or {}
    classification = evaluation.get("failure_classification") or details.get(
        "failure_classification"
    )
    classification = classification if isinstance(classification, dict) else {}
    category = evaluation.get("failure_category") or classification.get("category")
    codes = [str(code) for code in classification.get("codes") or []]
    if category and category not in codes:
        codes.insert(0, str(category))
    return (str(category) if category else None, codes)


def evidence_bucket(arm: dict[str, Any]) -> str:
    codes = set(arm["failure_codes"])
    category = arm.get("failure_category") or ""
    signals = arm.get("signals") or {}
    indicators = set(arm.get("raw_log_indicators") or [])
    groups: set[str] = set()

    if category.startswith("E") or any(code.startswith("E") for code in codes):
        groups.add("site_or_access_environment")
    if signals.get("network_navigation") or indicators:
        groups.add("site_or_access_environment")
    if category.startswith("M") or any(code.startswith("M") for code in codes):
        groups.add("agent_reasoning_or_evidence")
    if category.startswith("H") or any(code.startswith("H") for code in codes):
        groups.add("runtime_or_harness")
    if arm.get("agent_done") == "error" or signals.get("session_create"):
        groups.add("runtime_or_harness")
    if arm.get("agent_done") == "timeout" or signals.get("timeout"):
        groups.add("timeout_limit")

    if not groups:
        return "unresolved"
    if len(groups) > 1:
        return "mixed"
    return next(iter(groups))


def arm_record(
    task_summary: dict[str, Any],
    evaluation: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    details = evaluation.get("evaluation_details") or {}
    category, codes = failure_metadata(evaluation)
    answer = result.get("answer")
    return {
        "success": bool(task_summary.get("success")),
        "judge_score": details.get("score"),
        "agent_done": result.get("agent_done"),
        "agent_success": result.get("agent_success"),
        "env_status": result.get("env_status"),
        "steps": task_summary.get("steps"),
        "e2e_seconds": task_summary.get("e2e_seconds"),
        "signals": task_summary.get("signals") or {},
        "failure_category": category,
        "failure_codes": codes,
        "error_signature": error_signature(result),
        "raw_log_indicators": raw_log_indicators(result),
        "answer_length": len(answer or ""),
        "answer_sha256": hashlib.sha256((answer or "").encode()).hexdigest(),
    }


def paired_task_ids(
    dataset: dict[str, dict[str, Any]],
    lexmount_summary: dict[str, Any],
    local_summary: dict[str, Any],
    lexmount_evaluations: dict[str, dict[str, Any]],
    local_evaluations: dict[str, dict[str, Any]],
    lexmount_results: dict[str, dict[str, Any]],
    local_results: dict[str, dict[str, Any]],
) -> list[str]:
    lexmount_summary_ids = set(summary_task_records(lexmount_summary, "lexmount summary"))
    local_summary_ids = set(summary_task_records(local_summary, "local summary"))
    if lexmount_summary_ids != local_summary_ids:
        lexmount_only = sorted(lexmount_summary_ids - local_summary_ids, key=int)
        local_only = sorted(local_summary_ids - lexmount_summary_ids, key=int)
        raise ValueError(
            "summary task coverage differs: "
            f"lexmount only: {','.join(lexmount_only) or 'none'}; "
            f"local only: {','.join(local_only) or 'none'}"
        )
    summary_ids = lexmount_summary_ids
    sources = {
        "dataset": dataset,
        "lexmount evaluations": lexmount_evaluations,
        "local evaluations": local_evaluations,
        "lexmount results": lexmount_results,
        "local results": local_results,
    }
    missing = {
        name: sorted(summary_ids - set(records), key=int)
        for name, records in sources.items()
        if summary_ids - set(records)
    }
    if missing:
        detail = "; ".join(f"{name}: {','.join(ids)}" for name, ids in missing.items())
        raise ValueError(f"paired task coverage is incomplete: {detail}")
    return sorted(summary_ids, key=int)


def audit_paired_runs(
    dataset: dict[str, dict[str, Any]],
    lexmount_summary: dict[str, Any],
    local_summary: dict[str, Any],
    lexmount_evaluations: dict[str, dict[str, Any]],
    local_evaluations: dict[str, dict[str, Any]],
    lexmount_results: dict[str, dict[str, Any]],
    local_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lexmount_tasks = summary_task_records(lexmount_summary, "lexmount summary")
    local_tasks = summary_task_records(local_summary, "local summary")
    task_ids = paired_task_ids(
        dataset,
        lexmount_summary,
        local_summary,
        lexmount_evaluations,
        local_evaluations,
        lexmount_results,
        local_results,
    )
    outcome_counts: Counter[str] = Counter()
    bucket_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    category_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    target_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    error_signature_counts: dict[str, Counter[str]] = {
        "lexmount_only": Counter(),
        "local_only": Counter(),
    }
    discordant: list[dict[str, Any]] = []

    for task_id in task_ids:
        lex_success = bool(lexmount_tasks[task_id]["success"])
        local_success = bool(local_tasks[task_id]["success"])
        if lex_success and local_success:
            outcome = "both_success"
        elif lex_success:
            outcome = "lexmount_only"
        elif local_success:
            outcome = "local_only"
        else:
            outcome = "both_failed"
        outcome_counts[outcome] += 1
        if outcome not in bucket_counts:
            continue

        metadata = dataset[task_id]
        lex_arm = arm_record(
            lexmount_tasks[task_id],
            lexmount_evaluations[task_id],
            lexmount_results[task_id],
        )
        local_arm = arm_record(
            local_tasks[task_id],
            local_evaluations[task_id],
            local_results[task_id],
        )
        loser = local_arm if outcome == "lexmount_only" else lex_arm
        winner_result = lexmount_results[task_id] if outcome == "lexmount_only" else local_results[
            task_id
        ]
        loser_result = local_results[task_id] if outcome == "lexmount_only" else lexmount_results[
            task_id
        ]
        bucket = evidence_bucket(loser)
        threshold = int(metadata.get("score_threshold") or 0)
        loser_score = loser.get("judge_score")
        similarity = answer_similarity(winner_result.get("answer"), loser_result.get("answer"))
        near_threshold = (
            loser_score is not None and threshold > 0 and 0 <= threshold - loser_score <= 10
        )
        high_answer_similarity = similarity is not None and similarity >= 0.75
        bucket_counts[outcome][bucket] += 1
        category_counts[outcome][loser.get("failure_category") or "NONE"] += 1
        error_signature_counts[outcome][loser.get("error_signature") or "none"] += 1
        target_counts[outcome][str(metadata.get("target_website") or "unknown")] += 1
        discordant.append(
            {
                "task_id": task_id,
                "outcome": outcome,
                "query": metadata.get("query"),
                "target_website": metadata.get("target_website"),
                "domain": metadata.get("domain"),
                "language": metadata.get("language"),
                "website_region": metadata.get("website_region"),
                "task_type": metadata.get("task_type"),
                "difficulty": metadata.get("difficulty"),
                "score_threshold": threshold,
                "answer_similarity": similarity,
                "near_threshold_loser": near_threshold,
                "high_answer_similarity": high_answer_similarity,
                "evidence_bucket": bucket,
                "lexmount": lex_arm,
                "local": local_arm,
            }
        )

    return {
        "schema_version": 1,
        "paired_tasks": len(task_ids),
        "outcomes": dict(outcome_counts),
        "discordant_tasks": len(discordant),
        "loser_evidence_buckets": {
            outcome: dict(counts) for outcome, counts in bucket_counts.items()
        },
        "loser_primary_failure_categories": {
            outcome: dict(counts) for outcome, counts in category_counts.items()
        },
        "loser_error_signatures": {
            outcome: dict(counts) for outcome, counts in error_signature_counts.items()
        },
        "judge_sensitivity_flags": {
            "near_threshold_loser": sum(item["near_threshold_loser"] for item in discordant),
            "high_answer_similarity": sum(
                item["high_answer_similarity"] for item in discordant
            ),
            "both": sum(
                item["near_threshold_loser"] and item["high_answer_similarity"]
                for item in discordant
            ),
        },
        "top_targets": {
            outcome: dict(counts.most_common(10)) for outcome, counts in target_counts.items()
        },
        "discordant": discordant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit paired browser benchmark trajectories")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lexmount-summary", type=Path, required=True)
    parser.add_argument("--local-summary", type=Path, required=True)
    parser.add_argument("--lexmount-run", type=Path, required=True)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--lexmount-eval", type=Path, required=True)
    parser.add_argument("--local-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_paired_runs(
        load_dataset(args.dataset),
        load_json_object(args.lexmount_summary),
        load_json_object(args.local_summary),
        load_evaluations(args.lexmount_eval),
        load_evaluations(args.local_eval),
        load_results(args.lexmount_run),
        load_results(args.local_run),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
