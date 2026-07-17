from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "scripts" / "aggregate_webvoyager_posttrain_repeats.py"


def load_script_module() -> types.ModuleType:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        module_name = "aggregate_webvoyager_posttrain_repeats_for_test"
        spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def manifest(
    *, backend: str, seed: int, temperature: float = 1.0, revision: str = "revision"
) -> dict:
    return {
        "backend": backend,
        "protocol": "webvoyager-posttrain-v1",
        "schema_version": 1,
        "tasks": "/tmp/common-five.jsonl",
        "tasks_sha256": "tasks-sha",
        "selected_tasks": 2,
        "evaluator": {"repository_revision": revision, "script_sha256": "script-sha"},
        "generation": {"seed_base": seed, "temperature": temperature, "top_p": 1.0},
        "judge": {"mode": "training", "model": "gpt-5.5", "temperature": None},
        "model": {"id": "step150", "safetensors_sha256": "model-sha"},
        "browser": {"dom_backend": "cdp", "lexmount_official_proxy": False},
    }


def record(task_id: str, verdict: str) -> dict:
    return {
        "task": {"task_id": task_id, "website": "example", "split": "in_train"},
        "status": "completed",
        "final_answer_status": "complete",
        "judge": {"status": "ok", "verdict": verdict, "reward": 1.0 if verdict == "yes" else 0.0},
        "guard": {
            "infrastructure_failures": 0,
            "policy_failures": 0,
            "timeouts": 0,
            "termination_reason": "",
        },
        "events": [],
        "wall_seconds": 1.0,
    }


def write_arm(
    path: Path,
    *,
    backend: str,
    seed: int,
    rows: list[dict],
    temperature: float = 1.0,
    revision: str = "revision",
) -> None:
    path.mkdir(parents=True)
    (path / "run_manifest.json").write_text(
        json.dumps(
            manifest(backend=backend, seed=seed, temperature=temperature, revision=revision)
        ),
        encoding="utf-8",
    )
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_run(
    path: Path,
    *,
    seed: int,
    lexmount_rows: list[dict],
    local_rows: list[dict],
    temperature: float = 1.0,
    revision: str = "revision",
) -> None:
    write_arm(
        path / "lexmount",
        backend="lexmount",
        seed=seed,
        rows=lexmount_rows,
        temperature=temperature,
        revision=revision,
    )
    write_arm(
        path / "local",
        backend="local",
        seed=seed,
        rows=local_rows,
        temperature=temperature,
        revision=revision,
    )


def test_aggregate_keeps_repeated_observations_separate_from_tasks(tmp_path: Path) -> None:
    module = load_script_module()
    first = tmp_path / "seed-1"
    second = tmp_path / "seed-2"
    write_run(
        first,
        seed=1,
        lexmount_rows=[record("task-1", "yes"), record("task-2", "no")],
        local_rows=[record("task-1", "no"), record("task-2", "no")],
    )
    write_run(
        second,
        seed=2,
        lexmount_rows=[record("task-1", "yes"), record("task-2", "no")],
        local_rows=[record("task-1", "yes"), record("task-2", "no")],
    )

    aggregate = module.aggregate_repeats([first, second])

    assert aggregate["repeat_control_contract"]["matches"] is True
    assert aggregate["distinct_tasks"] == 2
    assert aggregate["repeat_runs"] == 2
    assert aggregate["paired_observations"] == 4
    assert aggregate["paired_quality"] == {
        "eligible_observations": 4,
        "outcomes": {"both_no": 2, "both_success": 1, "lexmount_only_success": 1},
        "lexmount_successes": 2,
        "local_successes": 1,
        "lexmount_success_rate": 0.5,
        "local_success_rate": 0.25,
        "lexmount_minus_local_success_rate": 0.25,
    }
    assert aggregate["per_task"][0]["task_id"] == "task-1"
    assert aggregate["per_task"][0]["observations"] == 2
    assert aggregate["per_task"][0]["quality_judge_outcomes"] == {
        "both_success": 1,
        "lexmount_only_success": 1,
    }


def test_aggregate_rejects_different_task_coverage(tmp_path: Path) -> None:
    module = load_script_module()
    first = tmp_path / "seed-1"
    second = tmp_path / "seed-2"
    write_run(
        first,
        seed=1,
        lexmount_rows=[record("task-1", "no")],
        local_rows=[record("task-1", "no")],
    )
    write_run(
        second,
        seed=2,
        lexmount_rows=[record("task-2", "no")],
        local_rows=[record("task-2", "no")],
    )

    with pytest.raises(ValueError, match="repeated task coverage differs"):
        module.aggregate_repeats([first, second])


def test_aggregate_flags_non_seed_control_difference(tmp_path: Path) -> None:
    module = load_script_module()
    first = tmp_path / "seed-1"
    second = tmp_path / "seed-2"
    rows = [record("task-1", "no")]
    write_run(first, seed=1, lexmount_rows=rows, local_rows=rows)
    write_run(second, seed=2, lexmount_rows=rows, local_rows=rows, temperature=0.7)

    aggregate = module.aggregate_repeats([first, second])

    assert aggregate["repeat_control_contract"]["matches"] is False
    assert set(aggregate["repeat_control_contract"]["control_differences"]) == {str(second)}
    assert set(aggregate["repeat_control_contract"]["control_differences"][str(second)]) == {
        "generation"
    }


def test_aggregate_allows_unrelated_repository_revision_change(tmp_path: Path) -> None:
    module = load_script_module()
    first = tmp_path / "seed-1"
    second = tmp_path / "seed-2"
    rows = [record("task-1", "no")]
    write_run(first, seed=1, lexmount_rows=rows, local_rows=rows, revision="first")
    write_run(second, seed=2, lexmount_rows=rows, local_rows=rows, revision="second")

    aggregate = module.aggregate_repeats([first, second])

    assert aggregate["repeat_control_contract"]["matches"] is True
