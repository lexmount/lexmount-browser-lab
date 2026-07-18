from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "scripts" / "compare_webvoyager_posttrain_models.py"


def load_script_module() -> types.ModuleType:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        module_name = "compare_webvoyager_posttrain_models_for_test"
        spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def manifest(*, model_id: str, model_sha: str) -> dict:
    return {
        "backend": "lexmount",
        "protocol": "webvoyager-posttrain-v1",
        "schema_version": 1,
        "tasks": "/tmp/smoke.jsonl",
        "tasks_sha256": "tasks-sha",
        "selected_tasks": 2,
        "evaluator": {"repository_revision": "revision", "script_sha256": "script-sha"},
        "generation": {"seed_base": 7, "temperature": 1.0},
        "judge": {"mode": "training", "model": "gpt-5.5", "temperature": None},
        "model": {"id": model_id, "safetensors_sha256": model_sha},
        "browser": {"dom_backend": "cdp", "lexmount_official_proxy": False},
    }


def record(task_id: str, *, verdict: str, infrastructure_failures: int = 0) -> dict:
    return {
        "task": {"task_id": task_id, "website": "example", "split": "smoke"},
        "status": "completed",
        "final_answer_status": "complete",
        "judge": {"status": "ok", "verdict": verdict, "reward": 1.0 if verdict == "yes" else 0.0},
        "guard": {
            "infrastructure_failures": infrastructure_failures,
            "policy_failures": 0,
            "timeouts": 0,
            "termination_reason": "",
        },
        "events": [],
    }


def write_run(path: Path, *, model_id: str, model_sha: str, rows: list[dict]) -> None:
    path.mkdir()
    (path / "run_manifest.json").write_text(
        json.dumps(manifest(model_id=model_id, model_sha=model_sha)), encoding="utf-8"
    )
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_checkpoint_comparison_allows_model_difference_but_not_control_difference(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    base_dir = tmp_path / "base"
    trained_dir = tmp_path / "trained"
    write_run(
        base_dir,
        model_id="base",
        model_sha="base-sha",
        rows=[
            record("task-1", verdict="no"),
            record("task-2", verdict="no", infrastructure_failures=1),
        ],
    )
    write_run(
        trained_dir,
        model_id="trained",
        model_sha="trained-sha",
        rows=[record("task-1", verdict="yes"), record("task-2", verdict="yes")],
    )

    comparison = module.compare_checkpoints(base_dir, trained_dir)

    assert comparison["control_contract"] == {"matches": True, "differences": {}}
    assert comparison["arms"]["base"]["quality_eligible"] == 1
    assert comparison["arms"]["trained"]["quality_eligible"] == 2
    assert comparison["paired_quality"] == {
        "eligible_tasks": 1,
        "outcomes": {"trained_only_success": 1},
        "base_successes": 0,
        "trained_successes": 1,
        "trained_minus_base_success_rate": 1.0,
    }
