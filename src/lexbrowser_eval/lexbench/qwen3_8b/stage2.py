#!/usr/bin/env python3
"""Run LexBench Stage 2 with five isolated official-evaluator shards.

This wrapper does not modify the official checkout.  Each shard calls the
official LexBench evaluator directly (scoring only), then the wrapper validates
and atomically merges all records before invoking the official failure
classification and summary helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

RESULTS_NAME = "task_gpt-5.4_per_task_threshold_stepwise_eval_results.json"
SUMMARY_NAME = "task_gpt-5.4_per_task_threshold_stepwise_summary.json"


def task_sort_key(task_id: str) -> tuple[int, int | str]:
    return (0, int(task_id)) if task_id.isdigit() else (1, task_id)


def deterministic_shards(task_ids: list[str], workers: int) -> list[list[str]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    ordered = sorted(set(task_ids), key=task_sort_key)
    if len(ordered) != len(task_ids):
        raise ValueError("task IDs must be unique")
    shards = [[] for _ in range(workers)]
    for index, task_id in enumerate(ordered):
        shards[index % workers].append(task_id)
    return shards


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"non-object record at {path}:{line_number}")
            records.append(value)
    return records


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def official_imports(checkout: pathlib.Path):
    sys.path.insert(0, str(checkout))
    from browseruse_bench.cli.eval import (  # type: ignore
        _merge_manifest_into_summary,
        refresh_summary_failure_stats,
        run_failure_classification,
    )
    from browseruse_bench.eval.base import EvaluatorArgs  # type: ignore
    from browseruse_bench.eval.lexbench_browser.evaluator import (  # type: ignore
        LexBenchBrowserEvaluator,
    )
    from browseruse_bench.utils import (  # type: ignore
        DataSource,
        load_config_file,
        load_evaluation_model,
    )

    return {
        "EvaluatorArgs": EvaluatorArgs,
        "LexBenchBrowserEvaluator": LexBenchBrowserEvaluator,
        "DataSource": DataSource,
        "load_config_file": load_config_file,
        "load_evaluation_model": load_evaluation_model,
        "run_failure_classification": run_failure_classification,
        "refresh_summary_failure_stats": refresh_summary_failure_stats,
        "merge_manifest": _merge_manifest_into_summary,
    }


def judge_settings(checkout: pathlib.Path) -> tuple[str, str, str, float | None, int | None]:
    api_key = (
        os.environ.get("LEXBENCH_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()
    base_url = (
        os.environ.get("LEXBENCH_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    ).strip()
    configured_model = os.environ.get("LEXBENCH_JUDGE_MODEL", "gpt-5.4").strip()
    if configured_model != "gpt-5.4":
        raise RuntimeError(f"LEXBENCH_JUDGE_MODEL must be gpt-5.4, got {configured_model!r}")
    if not api_key or not base_url:
        raise RuntimeError("LexBench Judge API key and base URL are required")
    imports = official_imports(checkout)
    config = imports["load_config_file"](checkout / "config.yaml")
    eval_config = config.get("eval", {})
    return (
        "gpt-5.4",
        api_key,
        base_url,
        eval_config.get("temperature"),
        eval_config.get("max_tokens"),
    )


def build_evaluator(checkout: pathlib.Path, trajectories: pathlib.Path, output: pathlib.Path):
    imports = official_imports(checkout)
    model, api_key, base_url, temperature, max_tokens = judge_settings(checkout)
    extra: dict[str, Any] = {"eval_strategy": "stepwise", "force_download": False}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    args = imports["EvaluatorArgs"](
        benchmark="LexBench-Browser",
        model=model,
        api_key=api_key,
        base_url=base_url,
        trajectories_dir=trajectories,
        output_path=output,
        score_threshold=None,
        num_worker=1,
        temperature=temperature,
        split="All",
        data_source=imports["DataSource"].LOCAL,
        mode="LexBench-Browser_eval",
        force_reeval=False,
        extra=extra,
    )
    model_instance = imports["load_evaluation_model"](
        model, api_key, base_url, temperature=temperature
    )
    return imports["LexBenchBrowserEvaluator"](args, model_instance), imports


def source_tasks(run_dir: pathlib.Path) -> list[pathlib.Path]:
    tasks_dir = run_dir / "tasks"
    tasks = sorted(
        (path for path in tasks_dir.iterdir() if path.is_dir() and (path / "result.json").exists()),
        key=lambda path: task_sort_key(path.name),
    )
    if len(tasks) != 210:
        raise RuntimeError(f"expected 210 completed trajectories, found {len(tasks)}")
    return tasks


def prepare(
    checkout: pathlib.Path, run_dir: pathlib.Path, stage_dir: pathlib.Path, workers: int
) -> None:
    tasks = source_tasks(run_dir)
    canonical = run_dir / "tasks_eval_result" / RESULTS_NAME
    if canonical.exists():
        raise RuntimeError(f"formal Stage 2 already exists: {canonical}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    assignments = deterministic_shards([task.name for task in tasks], workers)
    task_map = {task.name: task for task in tasks}
    manifest = {
        "schema_version": 1,
        "model": "gpt-5.4",
        "strategy": "stepwise",
        "workers": workers,
        "official_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "official_tracked_status": subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=checkout,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "source_run_dir": str(run_dir),
        "source_results": {
            task.name: {
                "sha256": sha256_file(task / "result.json"),
                "mtime_ns": (task / "result.json").stat().st_mtime_ns,
            }
            for task in tasks
        },
        "assignments": assignments,
    }
    if manifest["official_tracked_status"]:
        raise RuntimeError("official checkout has tracked changes")
    atomic_write_json(stage_dir / "manifest.json", manifest)

    for index, task_ids in enumerate(assignments):
        shard_tasks = stage_dir / "shards" / f"shard_{index}" / "tasks"
        shard_tasks.mkdir(parents=True, exist_ok=True)
        for task_id in task_ids:
            link = shard_tasks / task_id
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(task_map[task_id], target_is_directory=True)


def score(checkout: pathlib.Path, trajectories: pathlib.Path, output: pathlib.Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    evaluator, _ = build_evaluator(checkout, trajectories, output)
    return evaluator.run()


def validate_shards(stage_dir: pathlib.Path) -> list[dict[str, Any]]:
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {task_id for shard in manifest["assignments"] for task_id in shard}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, assignment in enumerate(manifest["assignments"]):
        path = stage_dir / "shards" / f"shard_{index}" / "tasks_eval_result" / RESULTS_NAME
        shard_records = read_jsonl(path)
        shard_ids = {str(record.get("task_id")) for record in shard_records}
        if shard_ids != set(assignment):
            raise RuntimeError(
                f"shard {index} coverage mismatch: "
                f"expected={len(assignment)} actual={len(shard_ids)}"
            )
        for record in shard_records:
            task_id = str(record.get("task_id"))
            if task_id in seen:
                raise RuntimeError(f"duplicate task_id across shards: {task_id}")
            seen.add(task_id)
            records.append(record)
    if seen != expected or len(records) != 210:
        raise RuntimeError(f"merged coverage mismatch: unique={len(seen)} records={len(records)}")
    return sorted(records, key=lambda record: task_sort_key(str(record["task_id"])))


def merge_and_summarize(
    checkout: pathlib.Path, run_dir: pathlib.Path, stage_dir: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    records = validate_shards(stage_dir)
    output = run_dir / "tasks_eval_result"
    output.mkdir(parents=True, exist_ok=True)
    canonical = output / RESULTS_NAME
    atomic_write_jsonl(canonical, records)

    evaluator, imports = build_evaluator(checkout, run_dir / "tasks", output)
    evaluator.load_tasks()
    canonical_records = evaluator._dedupe_results_file()
    evaluator._generate_summary(canonical_records)
    summary = output / SUMMARY_NAME
    model, _, base_url, _, _ = judge_settings(checkout)
    imports["merge_manifest"](
        summary,
        eval_mode="LexBench-Browser_eval",
        model=model,
        base_url=base_url,
        score_threshold=None,
        results_file=canonical,
        trajectories_dir=run_dir / "tasks",
        exit_code=0,
    )
    return canonical, summary


def classify(
    checkout: pathlib.Path,
    run_dir: pathlib.Path,
    canonical: pathlib.Path,
    summary: pathlib.Path,
    workers: int,
) -> int:
    imports = official_imports(checkout)
    model, api_key, base_url, temperature, _ = judge_settings(checkout)
    status = imports["run_failure_classification"](
        canonical,
        run_dir / "tasks",
        model,
        api_key,
        base_url,
        skip_existing=True,
        num_workers=workers,
        temperature=temperature,
    )
    if status == 0:
        imports["refresh_summary_failure_stats"](canonical, summary)
    return status


def run_all(args: argparse.Namespace) -> int:
    prepare(args.checkout, args.run_dir, args.stage_dir, args.workers)

    # Isolated one-task preflight: real official evaluator call, never formal output.
    manifest = json.loads((args.stage_dir / "manifest.json").read_text(encoding="utf-8"))
    dry_task_id = manifest["assignments"][0][0]
    dry_tasks = args.stage_dir / "dry_run" / "tasks"
    dry_tasks.mkdir(parents=True, exist_ok=True)
    (dry_tasks / dry_task_id).symlink_to(
        args.run_dir / "tasks" / dry_task_id, target_is_directory=True
    )
    if score(args.checkout, dry_tasks, args.stage_dir / "dry_run" / "tasks_eval_result") != 0:
        raise RuntimeError("isolated dry-run failed")
    dry_records = read_jsonl(args.stage_dir / "dry_run" / "tasks_eval_result" / RESULTS_NAME)
    if len(dry_records) != 1 or str(dry_records[0].get("task_id")) != dry_task_id:
        raise RuntimeError("isolated dry-run output mismatch")

    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    for index in range(args.workers):
        shard = args.stage_dir / "shards" / f"shard_{index}"
        log = (shard / "score.log").open("wb")
        command = [
            sys.executable,
            "-m",
            "lexbrowser_eval.lexbench.qwen3_8b.stage2",
            "score",
            "--checkout",
            str(args.checkout),
            "--trajectories",
            str(shard / "tasks"),
            "--output",
            str(shard / "tasks_eval_result"),
        ]
        processes.append(
            (index, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
        )
    failures: list[tuple[int, int]] = []
    for index, process, log in processes:
        status = process.wait()
        log.close()
        if status != 0:
            failures.append((index, status))
    if failures:
        raise RuntimeError(f"score shards failed: {failures}")

    canonical, summary = merge_and_summarize(args.checkout, args.run_dir, args.stage_dir)
    status = classify(args.checkout, args.run_dir, canonical, summary, args.workers)
    if status != 0:
        raise RuntimeError(f"failure classification exited {status}")
    atomic_write_json(
        args.stage_dir / "complete.json",
        {
            "model": "gpt-5.4",
            "workers": args.workers,
            "records": len(read_jsonl(canonical)),
            "results": str(canonical),
            "summary": str(summary),
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--checkout", type=pathlib.Path, required=True)
    run.add_argument("--run-dir", type=pathlib.Path, required=True)
    run.add_argument("--stage-dir", type=pathlib.Path, required=True)
    run.add_argument("--workers", type=int, default=5)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--checkout", type=pathlib.Path, required=True)
    score_parser.add_argument("--trajectories", type=pathlib.Path, required=True)
    score_parser.add_argument("--output", type=pathlib.Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "score":
        return score(args.checkout, args.trajectories, args.output)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
