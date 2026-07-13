#!/usr/bin/env python3
"""Score completed LexBench stress trajectories with five global Judge workers."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, TypeVar

from . import stage2 as official_stage2

RESULTS_NAME = official_stage2.RESULTS_NAME
SUMMARY_NAME = official_stage2.SUMMARY_NAME
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class SourceTask:
    task_id: str
    task_dir: pathlib.Path
    result_path: pathlib.Path
    sha256: str
    mtime_ns: int
    payload: dict[str, Any]
    instance_key: str = ""

    @property
    def instance_key_suffix(self) -> str:
        return self.task_id


@dataclass(frozen=True)
class ReplicaInstance:
    backend: str
    cell: str
    target_concurrency: int
    replica_index: int
    run_dir: pathlib.Path
    stage_name: str
    tasks: tuple[SourceTask, ...]

    @property
    def replica_key(self) -> str:
        return f"{self.backend}/{self.cell}/r{self.replica_index:02d}"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: pathlib.Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cell_name(backend: str, cell: dict[str, Any]) -> str:
    explicit = str(cell.get("cell_name") or "").strip()
    if explicit:
        return explicit
    target = int(cell["target_concurrency"])
    if target in {20, 60, 100, 200, 500}:
        return f"{backend}_c{target}"
    return f"{backend}_capacity_probe_c{target}"


def discover_instances(
    rollout_summary: dict[str, Any],
) -> tuple[list[ReplicaInstance], list[dict[str, Any]]]:
    instances: list[ReplicaInstance] = []
    excluded: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for backend, cells in rollout_summary.get("cells", {}).items():
        for cell in cells:
            cell_name = _cell_name(str(backend), cell)
            target = int(cell["target_concurrency"])
            for replica_index, raw_run_dir in enumerate(cell.get("official_run_dirs", [])):
                run_dir = pathlib.Path(raw_run_dir)
                tasks: list[SourceTask] = []
                tasks_root = run_dir / "tasks"
                for task_dir in sorted(
                    (path for path in tasks_root.glob("*") if path.is_dir()),
                    key=lambda path: official_stage2.task_sort_key(path.name),
                ):
                    result = task_dir / "result.json"
                    if not result.is_file():
                        continue
                    try:
                        payload = json.loads(result.read_text(encoding="utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("non-object")
                    except Exception as exc:  # noqa: BLE001
                        excluded.append(
                            {
                                "backend": backend,
                                "cell": cell_name,
                                "replica_index": replica_index,
                                "task_id": task_dir.name,
                                "reason": f"invalid_json:{type(exc).__name__}",
                            }
                        )
                        continue
                    task_id = str(payload.get("task_id") or task_dir.name)
                    instance_key = f"{backend}/{cell_name}/r{replica_index:02d}/{task_id}"
                    if instance_key in seen_keys:
                        raise RuntimeError(f"duplicate stress instance key: {instance_key}")
                    seen_keys.add(instance_key)
                    tasks.append(
                        SourceTask(
                            task_id=task_id,
                            task_dir=task_dir,
                            result_path=result,
                            sha256=sha256_file(result),
                            mtime_ns=result.stat().st_mtime_ns,
                            payload=payload,
                            instance_key=instance_key,
                        )
                    )
                if tasks:
                    stage_name = f"{backend}__{cell_name}__r{replica_index:02d}"
                    instances.append(
                        ReplicaInstance(
                            backend=str(backend),
                            cell=cell_name,
                            target_concurrency=target,
                            replica_index=replica_index,
                            run_dir=run_dir,
                            stage_name=stage_name,
                            tasks=tuple(tasks),
                        )
                    )
    return instances, excluded


def prepare_stage(
    *,
    checkout: pathlib.Path,
    rollout_summary: dict[str, Any],
    rollout_summary_path: pathlib.Path,
    stage_dir: pathlib.Path,
    workers: int,
    official_commit: str,
) -> dict[str, Any]:
    if stage_dir.exists():
        raise RuntimeError(f"Stage 2 directory already exists: {stage_dir}")
    stage_dir.mkdir(parents=True)
    instances, excluded = discover_instances(rollout_summary)
    if not instances:
        raise RuntimeError("No valid stress trajectories to score")
    manifest_instances: list[dict[str, Any]] = []
    for instance in instances:
        instance_root = stage_dir / "instances" / instance.stage_name
        tasks_root = instance_root / "tasks"
        tasks_root.mkdir(parents=True)
        task_records: list[dict[str, Any]] = []
        for task in instance.tasks:
            (tasks_root / task.task_id).symlink_to(task.task_dir, target_is_directory=True)
            task_records.append(
                {
                    "task_id": task.task_id,
                    "result_path": str(task.result_path),
                    "sha256": task.sha256,
                    "mtime_ns": task.mtime_ns,
                    "stress_instance_key": f"{instance.replica_key}/{task.task_id}",
                }
            )
        manifest_instances.append(
            {
                "backend": instance.backend,
                "cell": instance.cell,
                "target_concurrency": instance.target_concurrency,
                "replica_index": instance.replica_index,
                "replica_key": instance.replica_key,
                "stage_name": instance.stage_name,
                "source_run_dir": str(instance.run_dir),
                "tasks": task_records,
            }
        )
    manifest = {
        "schema_version": 1,
        "model": "gpt-5.4",
        "strategy": "stepwise",
        "workers": workers,
        "official_commit": official_commit,
        "rollout_summary": str(rollout_summary_path),
        "rollout_summary_sha256": (
            sha256_file(rollout_summary_path) if rollout_summary_path.is_file() else None
        ),
        "valid_trajectory_count": sum(len(item.tasks) for item in instances),
        "excluded": excluded,
        "instances": manifest_instances,
    }
    atomic_json(stage_dir / "manifest.json", manifest)
    return manifest


def validate_sources_unchanged(manifest: dict[str, Any]) -> None:
    for instance in manifest["instances"]:
        for task in instance["tasks"]:
            path = pathlib.Path(task["result_path"])
            if (
                not path.is_file()
                or path.stat().st_mtime_ns != int(task["mtime_ns"])
                or sha256_file(path) != task["sha256"]
            ):
                raise RuntimeError(f"Source result changed: {path}")


def run_bounded(items: Sequence[T], *, max_workers: int, worker: Callable[[T], R]) -> list[R]:
    if not 1 <= max_workers <= 5:
        raise ValueError("max_workers must be between 1 and 5")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, item) for item in items]
        return [future.result() for future in futures]


def _steps(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    for key in ("steps", "step_count", "num_steps"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    history = payload.get("action_history")
    return float(len(history)) if isinstance(history, list) else None


def _e2e_seconds(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    for key in ("end_to_end_ms", "e2e_ms"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 1000.0
    for key in ("end_to_end_seconds", "e2e_seconds"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    value = payload.get("wall_clock_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def enrich_instance_records(
    instance: ReplicaInstance, records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record.get("task_id"))
        if task_id in by_id:
            raise RuntimeError(f"duplicate official task record: {task_id}")
        by_id[task_id] = dict(record)
    expected = {task.task_id for task in instance.tasks}
    if set(by_id) != expected:
        raise RuntimeError(
            f"official score coverage mismatch for {instance.replica_key}: "
            f"expected={len(expected)} actual={len(by_id)}"
        )
    source = {task.task_id: task for task in instance.tasks}
    enriched: list[dict[str, Any]] = []
    for task_id in sorted(expected, key=official_stage2.task_sort_key):
        record = dict(by_id[task_id])
        record.update(
            {
                "stress_instance_key": f"{instance.replica_key}/{task_id}",
                "stress_backend": instance.backend,
                "stress_cell": instance.cell,
                "stress_target_concurrency": instance.target_concurrency,
                "stress_replica_index": instance.replica_index,
                "stress_steps": _steps(source[task_id].payload),
                "stress_e2e_seconds": _e2e_seconds(source[task_id].payload),
            }
        )
        enriched.append(record)
    return enriched


def _passed(record: dict[str, Any]) -> bool:
    value = record.get("predicted_label")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"1", "true", "pass", "passed", "success"}


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    steps = [float(r["stress_steps"]) for r in records if r.get("stress_steps") is not None]
    e2e = [
        float(r["stress_e2e_seconds"]) for r in records if r.get("stress_e2e_seconds") is not None
    ]
    successful = sum(_passed(record) for record in records)
    return {
        "evaluated_instances": len(records),
        "successful_instances": successful,
        "success_rate_percent": round(100.0 * successful / len(records), 4) if records else 0.0,
        "avg_steps": round(fmean(steps), 4) if steps else None,
        "avg_e2e_seconds": round(fmean(e2e), 4) if e2e else None,
    }


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return official_stage2.read_jsonl(path)


def score_one(args: argparse.Namespace) -> int:
    return official_stage2.score(
        args.checkout, args.instance_root / "tasks", args.instance_root / "tasks_eval_result"
    )


def classify_one(args: argparse.Namespace) -> int:
    output = args.instance_root / "tasks_eval_result"
    return official_stage2.classify(
        args.checkout,
        args.instance_root,
        output / RESULTS_NAME,
        output / SUMMARY_NAME,
        1,
    )


def run_all(args: argparse.Namespace) -> int:
    if args.workers != 5:
        raise RuntimeError("Stress Stage 2 requires exactly five global workers")
    official_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.checkout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    rollout = json.loads(args.rollout_summary.read_text(encoding="utf-8"))
    manifest = prepare_stage(
        checkout=args.checkout,
        rollout_summary=rollout,
        rollout_summary_path=args.rollout_summary,
        stage_dir=args.stage_dir,
        workers=args.workers,
        official_commit=official_commit,
    )
    instances, _ = discover_instances(rollout)
    by_name = {instance.stage_name: instance for instance in instances}

    def run_command(item: dict[str, Any], command: str) -> int:
        instance_root = args.stage_dir / "instances" / item["stage_name"]
        log = instance_root / f"{command}.log"
        with log.open("wb") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lexbrowser_eval.lexbench.qwen3_8b.stress_stage2",
                    command,
                    "--checkout",
                    str(args.checkout),
                    "--instance-root",
                    str(instance_root),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{command} failed for {item['stage_name']}: {completed.returncode}")
        return completed.returncode

    run_bounded(
        manifest["instances"],
        max_workers=args.workers,
        worker=lambda item: run_command(item, "score-one"),
    )
    validate_sources_unchanged(manifest)
    run_bounded(
        manifest["instances"],
        max_workers=args.workers,
        worker=lambda item: run_command(item, "classify-one"),
    )
    validate_sources_unchanged(manifest)

    enriched: list[dict[str, Any]] = []
    for item in manifest["instances"]:
        instance = by_name[item["stage_name"]]
        result_path = (
            args.stage_dir / "instances" / item["stage_name"] / "tasks_eval_result" / RESULTS_NAME
        )
        enriched.extend(enrich_instance_records(instance, _read_jsonl(result_path)))
    if len({record["stress_instance_key"] for record in enriched}) != len(enriched):
        raise RuntimeError("duplicate enriched stress instance keys")
    aggregate = aggregate_records(enriched)
    aggregate["model"] = "gpt-5.4"
    aggregate["workers"] = args.workers
    aggregate["valid_trajectory_count"] = manifest["valid_trajectory_count"]
    aggregate["planned_instances_by_cell"] = {
        f"{backend}/{_cell_name(str(backend), cell)}": int(cell["planned_instances"])
        for backend, cells in rollout.get("cells", {}).items()
        for cell in cells
    }
    atomic_jsonl(args.stage_dir / "stress_gpt-5.4_stepwise_results.jsonl", enriched)
    atomic_json(args.stage_dir / "aggregate.json", aggregate)
    atomic_json(
        args.stage_dir / "complete.json",
        {
            "model": "gpt-5.4",
            "workers": args.workers,
            "evaluated_instances": len(enriched),
            "aggregate": str(args.stage_dir / "aggregate.json"),
            "results": str(args.stage_dir / "stress_gpt-5.4_stepwise_results.jsonl"),
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--checkout", type=pathlib.Path, required=True)
    run.add_argument("--rollout-summary", type=pathlib.Path, required=True)
    run.add_argument("--stage-dir", type=pathlib.Path, required=True)
    run.add_argument("--workers", type=int, default=5)
    for name in ("score-one", "classify-one"):
        child = sub.add_parser(name)
        child.add_argument("--checkout", type=pathlib.Path, required=True)
        child.add_argument("--instance-root", type=pathlib.Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "score-one":
        return score_one(args)
    if args.command == "classify-one":
        return classify_one(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
