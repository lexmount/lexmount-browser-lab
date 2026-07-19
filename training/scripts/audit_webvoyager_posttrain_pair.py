#!/usr/bin/env python3
"""Audit a paired WebVoyager post-training evaluation without conflating failures.

The post-training runner writes one JSONL record per task.  A judge verdict is
only a model-quality observation when the browser completed the setup and no
infrastructure failure occurred during the trajectory.  This utility turns two
same-task runs into a compact, credential-free audit with those denominators
kept separate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ERROR_CODE_PATTERN = re.compile(r"\b(ERROR_[A-Z_]+)\b")
RATE_LIMIT_PAGE_PATTERN = re.compile(
    r"\b(?:secondary rate limit|too many requests)\b", re.IGNORECASE
)
CONTRACT_KEYS = (
    "protocol",
    "schema_version",
    "tasks",
    "tasks_sha256",
    "selected_tasks",
    "evaluator",
    "generation",
    "judge",
    "model",
    "browser",
    "egress",
)
RESOURCE_METRICS = (
    "cpu_cores_mean",
    "external_cpu_cores_mean",
    "combined_cpu_cores_mean",
    "pss_gib",
    "chrome_pss_gib",
    "external_pss_gib",
    "combined_pss_gib",
    "gpu_utilization_percent_mean",
    "gpu_memory_mib_mean",
    "gpu_power_w_mean",
    "host_available_gib_min",
    "vllm_running_nonzero_fraction",
    "vllm_waiting_nonzero_fraction",
)
INTENTIONAL_BACKEND_BROWSER_DIFFERENCES = (
    "local_disable_automation_controlled",
)
SHARED_EGRESS_PROXY_BROWSER_DIFFERENCES = (
    "local_proxy_configured",
    "lexmount_external_proxy_configured",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def task_id(record: Mapping[str, Any]) -> str:
    task = record.get("task")
    if isinstance(task, Mapping):
        value = task.get("task_id") or task.get("id")
        if value:
            return str(value)
    value = record.get("task_id")
    if value:
        return str(value)
    raise ValueError("result record is missing task.task_id")


def index_results(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        identifier = task_id(record)
        if identifier in indexed:
            raise ValueError(f"{path}: duplicate task id {identifier}")
        indexed[identifier] = record
    return indexed


def event_error_codes(record: Mapping[str, Any]) -> list[str]:
    codes: set[str] = set()
    for event in record.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        codes.update(ERROR_CODE_PATTERN.findall(str(event.get("result") or "")))
    return sorted(codes)


def semantic_infrastructure_error_codes(record: Mapping[str, Any]) -> list[str]:
    """Detect known target-side block pages in legacy trajectory records.

    Older runs did not classify GitHub's rendered secondary-rate-limit page as
    an infrastructure error.  The page makes the task unavailable even though
    navigation itself completed, so it must not enter the quality denominator.
    """

    for event in record.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        if RATE_LIMIT_PAGE_PATTERN.search(str(event.get("result") or "")):
            return ["ERROR_INFRASTRUCTURE_RATE_LIMIT_PAGE"]
    return []


def normalize_arm(record: Mapping[str, Any]) -> dict[str, Any]:
    task = record.get("task") if isinstance(record.get("task"), Mapping) else {}
    guard = record.get("guard") if isinstance(record.get("guard"), Mapping) else {}
    judge = record.get("judge") if isinstance(record.get("judge"), Mapping) else {}
    status = str(record.get("status") or "unknown")
    infrastructure_failures = int(guard.get("infrastructure_failures") or 0)
    policy_failures = int(guard.get("policy_failures") or 0)
    timeouts = int(guard.get("timeouts") or 0)
    setup_or_runner_error = status != "completed"
    semantic_infrastructure_codes = semantic_infrastructure_error_codes(record)
    infrastructure = (
        setup_or_runner_error
        or infrastructure_failures > 0
        or timeouts > 0
        or bool(semantic_infrastructure_codes)
    )
    judge_status = str(judge.get("status") or "missing")
    verdict = judge.get("verdict") if judge.get("verdict") in {"yes", "no"} else None
    quality_eligible = status == "completed" and not infrastructure and judge_status == "ok"

    return {
        "task_id": task_id(record),
        "website": str(task.get("website") or ""),
        "split": str(task.get("split") or ""),
        "status": status,
        "final_answer_status": str(record.get("final_answer_status") or "unknown"),
        "judge_status": judge_status,
        "judge_verdict": verdict,
        "judge_reward": judge.get("reward"),
        "quality_eligible": quality_eligible,
        "setup_or_runner_error": setup_or_runner_error,
        "infrastructure_failure": infrastructure,
        "policy_failure": policy_failures > 0,
        "timeout": timeouts > 0,
        "guard": {
            "infrastructure_failures": infrastructure_failures,
            "policy_failures": policy_failures,
            "timeouts": timeouts,
            "termination_reason": str(guard.get("termination_reason") or ""),
        },
        "event_error_codes": sorted(
            set(event_error_codes(record)) | set(semantic_infrastructure_codes)
        ),
        "wall_seconds": record.get("wall_seconds"),
    }


def arm_metrics(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks = len(arms)
    judged = [arm for arm in arms if arm["judge_verdict"] in {"yes", "no"}]
    eligible = [arm for arm in arms if arm["quality_eligible"]]
    successes = sum(arm["judge_verdict"] == "yes" for arm in arms)
    eligible_successes = sum(arm["judge_verdict"] == "yes" for arm in eligible)
    return {
        "tasks": tasks,
        "completed": sum(arm["status"] == "completed" for arm in arms),
        "judged": len(judged),
        "judge_successes": successes,
        "raw_success_rate": successes / tasks if tasks else None,
        "judged_success_rate": successes / len(judged) if judged else None,
        "quality_eligible": len(eligible),
        "quality_successes": eligible_successes,
        "quality_success_rate": eligible_successes / len(eligible) if eligible else None,
        "setup_or_runner_errors": sum(arm["setup_or_runner_error"] for arm in arms),
        "infrastructure_failures": sum(arm["infrastructure_failure"] for arm in arms),
        "policy_failures": sum(arm["policy_failure"] for arm in arms),
        "timeouts": sum(arm["timeout"] for arm in arms),
        "final_answer_statuses": dict(
            sorted(Counter(str(arm["final_answer_status"]) for arm in arms).items())
        ),
        "event_error_codes": dict(
            sorted(
                Counter(code for arm in arms for code in arm["event_error_codes"]).items()
            )
        ),
    }


def outcome(left: str | None, right: str | None) -> str:
    if left not in {"yes", "no"} or right not in {"yes", "no"}:
        return "unjudged"
    if left == "yes" and right == "yes":
        return "both_success"
    if left == "yes":
        return "lexmount_only_success"
    if right == "yes":
        return "local_only_success"
    return "both_no"


def resource_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "resource_summary.json"
    if not path.exists():
        return None
    payload = load_json(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    return {
        "return_code": payload.get("return_code"),
        "duration_seconds": payload.get("duration_seconds"),
        "sample_count": payload.get("sample_count"),
        "gpu_index": payload.get("gpu_index"),
        "metrics": {key: metrics[key] for key in RESOURCE_METRICS if key in metrics},
    }


def comparison_contract(
    lexmount_manifest: Mapping[str, Any], local_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    normalized_lexmount = dict(lexmount_manifest)
    normalized_local = dict(local_manifest)
    lexmount_browser = dict(lexmount_manifest.get("browser") or {})
    local_browser = dict(local_manifest.get("browser") or {})
    lexmount_egress = lexmount_manifest.get("egress")
    local_egress = local_manifest.get("egress")
    lexmount_egress_id = (
        str(lexmount_egress.get("equivalence_id") or "").strip()
        if isinstance(lexmount_egress, Mapping)
        else ""
    )
    local_egress_id = (
        str(local_egress.get("equivalence_id") or "").strip()
        if isinstance(local_egress, Mapping)
        else ""
    )
    shared_egress_proxy_wiring = (
        bool(lexmount_egress_id)
        and lexmount_egress_id == local_egress_id
        and bool(local_browser.get("local_proxy_configured"))
        and bool(lexmount_browser.get("lexmount_external_proxy_configured"))
    )
    intentional_browser_differences = list(INTENTIONAL_BACKEND_BROWSER_DIFFERENCES)
    if shared_egress_proxy_wiring:
        intentional_browser_differences.extend(SHARED_EGRESS_PROXY_BROWSER_DIFFERENCES)
    intentional_backend_differences = {
        f"browser.{key}": {"lexmount": lexmount_browser.get(key), "local": local_browser.get(key)}
        for key in intentional_browser_differences
        if lexmount_browser.get(key) != local_browser.get(key)
    }
    for key in intentional_browser_differences:
        lexmount_browser.pop(key, None)
        local_browser.pop(key, None)
    normalized_lexmount["browser"] = lexmount_browser
    normalized_local["browser"] = local_browser
    differences = {
        key: {"lexmount": normalized_lexmount.get(key), "local": normalized_local.get(key)}
        for key in CONTRACT_KEYS
        if normalized_lexmount.get(key) != normalized_local.get(key)
    }
    result = {"matches": not differences, "differences": differences}
    if shared_egress_proxy_wiring:
        result["shared_egress_equivalence_id"] = lexmount_egress_id
    if intentional_backend_differences:
        result["intentional_backend_differences"] = intentional_backend_differences
    return result


def audit_pair(lexmount_dir: Path, local_dir: Path) -> dict[str, Any]:
    lexmount_manifest = load_json(lexmount_dir / "run_manifest.json")
    local_manifest = load_json(local_dir / "run_manifest.json")
    lexmount_records = index_results(lexmount_dir / "results.jsonl")
    local_records = index_results(local_dir / "results.jsonl")
    lexmount_ids = set(lexmount_records)
    local_ids = set(local_records)
    if lexmount_ids != local_ids:
        raise ValueError(
            "paired task coverage differs: "
            f"lexmount_only={sorted(lexmount_ids - local_ids)}; "
            f"local_only={sorted(local_ids - lexmount_ids)}"
        )

    pairs: list[dict[str, Any]] = []
    for identifier in sorted(lexmount_ids):
        lexmount_arm = normalize_arm(lexmount_records[identifier])
        local_arm = normalize_arm(local_records[identifier])
        if (lexmount_arm["website"], lexmount_arm["split"]) != (
            local_arm["website"],
            local_arm["split"],
        ):
            raise ValueError(f"task metadata differs between arms for {identifier}")
        raw_outcome = outcome(lexmount_arm["judge_verdict"], local_arm["judge_verdict"])
        pair_eligible = bool(lexmount_arm["quality_eligible"] and local_arm["quality_eligible"])
        pairs.append(
            {
                "task_id": identifier,
                "website": lexmount_arm["website"],
                "split": lexmount_arm["split"],
                "quality_pair_eligible": pair_eligible,
                "raw_judge_outcome": raw_outcome,
                "quality_judge_outcome": raw_outcome if pair_eligible else "ineligible",
                "lexmount": lexmount_arm,
                "local": local_arm,
            }
        )

    lexmount_arms = [pair["lexmount"] for pair in pairs]
    local_arms = [pair["local"] for pair in pairs]
    quality_pairs = [pair for pair in pairs if pair["quality_pair_eligible"]]
    return {
        "schema_version": 1,
        "comparison_contract": comparison_contract(lexmount_manifest, local_manifest),
        "tasks": len(pairs),
        "arms": {"lexmount": arm_metrics(lexmount_arms), "local": arm_metrics(local_arms)},
        "paired_quality": {
            "eligible_tasks": len(quality_pairs),
            "outcomes": dict(
                sorted(Counter(pair["quality_judge_outcome"] for pair in quality_pairs).items())
            ),
            "lexmount_successes": sum(
                pair["lexmount"]["judge_verdict"] == "yes" for pair in quality_pairs
            ),
            "local_successes": sum(
                pair["local"]["judge_verdict"] == "yes" for pair in quality_pairs
            ),
        },
        "raw_judge_outcomes": dict(
            sorted(Counter(pair["raw_judge_outcome"] for pair in pairs).items())
        ),
        "resources": {
            "lexmount": resource_summary(lexmount_dir),
            "local": resource_summary(local_dir),
        },
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit matched WebVoyager post-training Lexmount/local runs."
    )
    parser.add_argument("--lexmount-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict-contract",
        action="store_true",
        help="fail instead of writing an audit when non-backend run controls differ",
    )
    args = parser.parse_args()
    audit = audit_pair(args.lexmount_dir, args.local_dir)
    if args.strict_contract and not audit["comparison_contract"]["matches"]:
        raise SystemExit("paired manifests differ; refusing a strict comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
