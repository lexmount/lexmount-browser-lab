"""Build a campaign-level Qwen3-8B LexBench report from official artifacts."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import tempfile
from collections.abc import Iterable
from typing import Any

from .official import resolve_output_marker
from .protocol import PROTOCOL

GIB = 1024**3
ROLLOUT_PHASES = {"ramp", "steady", "drain"}


def _atomic_write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _p95(values: Iterable[float]) -> float | None:
    items = sorted(values)
    if not items:
        return None
    index = max(0, (95 * len(items) + 99) // 100 - 1)
    return items[index]


def _float(row: dict[str, str], name: str) -> float | None:
    raw = row.get(name, "").strip()
    return float(raw) if raw else None


def summarize_resource_files(
    resource_path: pathlib.Path, gpu_path: pathlib.Path
) -> dict[str, float | int | None]:
    if not resource_path.is_file() or not gpu_path.is_file():
        return {"available": False}

    with resource_path.open(encoding="utf-8", newline="") as handle:
        resource_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("phase") in ROLLOUT_PHASES
            and int(float(row.get("started_instance_count") or 0)) > 0
        ]
    cpu = [value for row in resource_rows if (value := _float(row, "cpu_cores")) is not None]
    pss_rows = [
        row
        for row in resource_rows
        if _float(row, "process_tree_pss_bytes") is not None
        and (_float(row, "pss_sample_age_seconds") or 0.0) <= 15.0
    ]
    pss = [float(row["process_tree_pss_bytes"]) / GIB for row in pss_rows]
    chrome_pss = [float(row["chrome_pss_bytes"]) / GIB for row in pss_rows]
    memory_peak = [
        value / GIB
        for row in resource_rows
        if (value := _float(row, "memory_peak_bytes")) is not None
    ]

    with gpu_path.open(encoding="utf-8", newline="") as handle:
        gpu_sm = [
            float(row["gpu_sm_percent"])
            for row in csv.DictReader(handle)
            if row.get("phase") in ROLLOUT_PHASES and row.get("gpu_sm_percent", "").strip()
        ]
    gpu_sm_mean = _mean(gpu_sm)
    return {
        "available": True,
        "resource_samples": len(resource_rows),
        "gpu_samples": len(gpu_sm),
        "cpu_cores_mean": _mean(cpu),
        "pss_gib_mean": _mean(pss),
        "pss_gib_p95": _p95(pss),
        "chrome_pss_gib_mean": _mean(chrome_pss),
        "chrome_pss_gib_p95": _p95(chrome_pss),
        "cgroup_memory_peak_gib": max(memory_peak) if memory_peak else None,
        "gpu_sm_mean": gpu_sm_mean,
        "gpu_idle_mean": 100.0 - gpu_sm_mean if gpu_sm_mean is not None else None,
    }


def _quality_summary(
    checkout: pathlib.Path,
    results_root: pathlib.Path,
    campaign_id: str,
    backend: str,
    task_count: int,
) -> dict[str, Any]:
    concurrency = min(PROTOCOL.quality_concurrency, task_count)
    cell_name = f"quality_{backend}_c{concurrency}"
    if task_count != PROTOCOL.quality_task_count:
        cell_name += f"_n{task_count}"
    marker = (
        results_root
        / campaign_id
        / cell_name
        / "official_run_dir.txt"
    )
    if not marker.is_file():
        raise RuntimeError(f"quality output marker is missing: {marker}")
    run_dir = resolve_output_marker(checkout, marker)
    summary_path = (
        run_dir / "tasks_eval_result" / f"task_{PROTOCOL.judge_model}_per_task_threshold_"
        f"{PROTOCOL.judge_strategy}_summary.json"
    )
    summary = _read_json(summary_path)
    evaluation = summary.get("evaluation_config", {})
    if evaluation.get("model") != PROTOCOL.judge_model:
        raise RuntimeError(f"quality Judge model mismatch: {summary_path}")
    if evaluation.get("eval_strategy") != PROTOCOL.judge_strategy:
        raise RuntimeError(f"quality Judge strategy mismatch: {summary_path}")
    overall = summary["overall_statistics"]
    metrics = summary["metrics_statistics"]
    if overall["evaluated_tasks"] != task_count:
        raise RuntimeError(f"quality coverage mismatch: {summary_path}")
    return {
        "backend": backend,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "evaluated_tasks": overall["evaluated_tasks"],
        "successful_tasks": overall["successful_tasks"],
        "success_rate_percent": overall["success_rate"],
        "avg_steps": metrics["steps"]["mean"],
        "avg_e2e_seconds": metrics["end_to_end_ms"]["mean"] / 1000.0,
    }


def _cell_resource_index(stress_root: pathlib.Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in sorted((stress_root / "cells").glob("*/cell_summary.json")):
        cell = _read_json(path)
        key = (cell["backend"], int(cell["target_concurrency"]), cell["completed_at"])
        monitor = path.parent / "monitor"
        indexed[key] = {
            "cell_name": path.parent.name,
            "resource_files": {
                "resource_samples": str(monitor / "resource_samples.csv"),
                "gpu_samples": str(monitor / "gpu_samples.csv"),
            },
            "resources": summarize_resource_files(
                monitor / "resource_samples.csv", monitor / "gpu_samples.csv"
            ),
        }
    return indexed


def _stress_summary(stress_root: pathlib.Path) -> dict[str, Any]:
    rollout_path = stress_root / "rollout_summary.json"
    aggregate_path = stress_root / "stage2_gpt54_c5" / "aggregate.json"
    rollout = _read_json(rollout_path)
    aggregate = _read_json(aggregate_path)
    if not rollout.get("rollout_complete"):
        raise RuntimeError(f"stress rollout is incomplete: {rollout_path}")
    if aggregate.get("model") != PROTOCOL.judge_model:
        raise RuntimeError(f"stress Judge model mismatch: {aggregate_path}")
    resource_index = _cell_resource_index(stress_root)
    cells: list[dict[str, Any]] = []
    for backend, entries in rollout.get("cells", {}).items():
        for entry in entries:
            key = (backend, int(entry["target_concurrency"]), entry["completed_at"])
            cells.append({**entry, **resource_index.get(key, {})})
    return {
        "rollout_summary_path": str(rollout_path),
        "aggregate_path": str(aggregate_path),
        "maximum_sustainable_concurrency": rollout["maximum_sustainable_concurrency"],
        "first_failed_concurrency": rollout["first_failed_concurrency"],
        "quality": aggregate,
        "cells": cells,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# LexBench Qwen3-8B campaign: {payload['campaign_id']}",
        "",
        "## 质量指标",
        "",
        "| Backend | Success | Avg steps | Avg e2e(s) |",
        "|---|---:|---:|---:|",
    ]
    for item in payload.get("quality", []):
        lines.append(
            f"| {item['backend']} | {item['success_rate_percent']:.2f}% "
            f"({item['successful_tasks']}/{item['evaluated_tasks']}) | "
            f"{item['avg_steps']:.2f} | {item['avg_e2e_seconds']:.2f} |"
        )
    stress = payload.get("stress")
    if stress:
        quality = stress["quality"]
        lines.extend(
            [
                "",
                "## 压力测试总览",
                "",
                f"Stage 2: {quality['successful_instances']}/{quality['evaluated_instances']} "
                f"({quality['success_rate_percent']:.2f}%), Judge={quality['model']}.",
                "",
                "| Backend | 并发 | 结果 | 完成 | CPU cores | PSS mean/P95 (GiB) | "
                "Chrome PSS mean/P95 (GiB) | GPU idle | 吞吐(task/h) |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in stress["cells"]:
            resources = cell.get("resources", {})
            status = "通过" if cell["capacity_pass"] else cell.get("capacity_failure_reason")
            lines.append(
                f"| {cell['backend']} | {cell['target_concurrency']} | {status} | "
                f"{cell['terminal_instances']}/{cell['planned_instances']} | "
                f"{_fmt(resources.get('cpu_cores_mean'))} | "
                f"{_fmt(resources.get('pss_gib_mean'))}/{_fmt(resources.get('pss_gib_p95'))} | "
                f"{_fmt(resources.get('chrome_pss_gib_mean'))}/"
                f"{_fmt(resources.get('chrome_pss_gib_p95'))} | "
                f"{_fmt(resources.get('gpu_idle_mean'))}% | "
                f"{cell['throughput_task_per_hour']:.2f} |"
            )
        lines.extend(
            [
                "",
                "GPU 为5090整卡观察值，可能包含批准的外部负载，不用于容量判定。",
                "CPU/PSS仅统计已启动任务的ramp、steady、drain样本；PSS排除采样年龄超过15秒的记录。",
                "GPU idle按相同三个阶段的整卡GPU SM样本计算为100%-mean(SM)。",
            ]
        )
    lines.extend(
        [
            "",
            "机器可读数据见同目录 `campaign_report.json`；原始轨迹、Judge、CSV路径保存在该文件中。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_campaign_report(
    *,
    checkout: pathlib.Path,
    results_root: pathlib.Path,
    campaign_id: str,
    backends: tuple[str, ...],
    include_quality: bool,
    include_stress: bool,
    quality_task_count: int = PROTOCOL.quality_task_count,
) -> tuple[pathlib.Path, pathlib.Path]:
    campaign_root = results_root / campaign_id
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "protocol": {
            "official_commit": PROTOCOL.upstream_commit,
            "dataset": "LexBench-Browser/All",
            "dataset_sha256": PROTOCOL.quality_sha256,
            "agent_model": PROTOCOL.agent_model_id,
            "judge_model": PROTOCOL.judge_model,
            "judge_strategy": PROTOCOL.judge_strategy,
            "quality_task_count": quality_task_count,
        },
    }
    if include_quality:
        payload["quality"] = [
            _quality_summary(checkout, results_root, campaign_id, backend, quality_task_count)
            for backend in backends
        ]
    if include_stress:
        payload["stress"] = _stress_summary(campaign_root / "stress_process_attributed")
    json_path = campaign_root / "campaign_report.json"
    markdown_path = campaign_root / "campaign_report.md"
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(markdown_path, _markdown(payload))
    return json_path, markdown_path
