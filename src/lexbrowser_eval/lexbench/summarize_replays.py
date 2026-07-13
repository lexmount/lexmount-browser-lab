#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def summary_run_dir(summary: dict[str, Any]) -> Path:
    run_dir = summary.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        raise ValueError("summary is missing a valid 'run_dir'")
    return Path(run_dir)


def summary_task_records(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict):
        raise ValueError("summary is missing a valid 'per_task' object")
    for task_id, task in per_task.items():
        if not isinstance(task, dict):
            raise ValueError(f"summary task {task_id} is not a JSON object")
        if "success" not in task:
            raise ValueError(f"summary task {task_id} is missing 'success'")
    return per_task


def synthetic_evaluation_ids(summary: dict[str, Any]) -> set[str]:
    run_dir = summary_run_dir(summary)
    eval_paths = sorted((run_dir / "tasks_eval_result").glob("*_eval_results.json"))
    if not eval_paths:
        return set()
    synthetic: set[str] = set()
    eval_path = eval_paths[-1]
    for record in load_json_records(eval_path):
        details = record.get("evaluation_details") or {}
        benchmark = details.get("benchmark_details") or {}
        if benchmark.get("is_synthetic_failure"):
            task_id = record.get("task_id")
            if task_id in (None, ""):
                raise ValueError(f"{eval_path}: synthetic evaluation is missing task_id")
            synthetic.add(str(task_id))
    return synthetic


def arm_task_record(task_id: str, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for summary in summaries:
        task = summary_task_records(summary)[task_id]
        synthetic = task_id in synthetic_evaluation_ids(summary)
        runs.append(
            {
                "run_id": summary_run_dir(summary).name,
                "success": None if synthetic else bool(task["success"]),
                "judge_score": None if synthetic else task.get("judge_score"),
                "synthetic_judge_failure": synthetic,
                "agent_done": task.get("agent_done"),
                "signals": task.get("signals") or {},
            }
        )
    valid = [run for run in runs if not run["synthetic_judge_failure"]]
    successes = sum(bool(run["success"]) for run in valid)
    return {
        "valid_judgments": len(valid),
        "successes": successes,
        "pass_rate": round(successes / len(valid), 6) if valid else None,
        "runs": runs,
    }


def summarize_replays(
    lexmount_summaries: list[dict[str, Any]], local_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    if not lexmount_summaries or not local_summaries:
        raise ValueError("at least one summary is required for each arm")
    all_summaries = lexmount_summaries + local_summaries
    task_sets = [set(summary_task_records(summary)) for summary in all_summaries]
    shared_tasks = set.intersection(*task_sets)
    if not shared_tasks:
        raise ValueError("summaries do not share any task ids")
    task_ids = sorted(shared_tasks, key=int)
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        tasks.append(
            {
                "task_id": task_id,
                "lexmount": arm_task_record(task_id, lexmount_summaries),
                "local": arm_task_record(task_id, local_summaries),
            }
        )

    def aggregate(arm: str) -> dict[str, Any]:
        valid = sum(task[arm]["valid_judgments"] for task in tasks)
        successes = sum(task[arm]["successes"] for task in tasks)
        synthetic = sum(
            run["synthetic_judge_failure"] for task in tasks for run in task[arm]["runs"]
        )
        return {
            "task_attempts": len(tasks) * len(tasks[0][arm]["runs"]) if tasks else 0,
            "valid_judgments": valid,
            "synthetic_judge_failures": synthetic,
            "successes": successes,
            "pass_rate": round(successes / valid, 6) if valid else None,
        }

    return {
        "schema_version": 1,
        "selected_tasks": len(tasks),
        "repetitions_per_arm": {
            "lexmount": len(lexmount_summaries),
            "local": len(local_summaries),
        },
        "aggregate": {"lexmount": aggregate("lexmount"), "local": aggregate("local")},
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize repeated targeted browser runs")
    parser.add_argument("--lexmount-summary", type=Path, action="append", required=True)
    parser.add_argument("--local-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(paths: list[Path]) -> list[dict[str, Any]]:
        return [load_json_object(path) for path in paths]

    summary = summarize_replays(load(args.lexmount_summary), load(args.local_summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
