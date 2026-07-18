from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "scripts" / "audit_webvoyager_posttrain_pair.py"


def load_script_module() -> types.ModuleType:
    module_name = "audit_webvoyager_posttrain_pair_for_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def manifest(*, backend: str, seed: int = 7) -> dict:
    return {
        "backend": backend,
        "protocol": "webvoyager-posttrain-v1",
        "schema_version": 1,
        "tasks": "/tmp/smoke.jsonl",
        "tasks_sha256": "tasks-sha",
        "selected_tasks": 2,
        "evaluator": {"repository_revision": "revision", "script_sha256": "script-sha"},
        "generation": {"seed_base": seed, "temperature": 1.0},
        "judge": {"mode": "training", "model": "gpt-5.5", "temperature": None},
        "model": {"id": "checkpoint", "safetensors_sha256": "model-sha"},
        "browser": {"dom_backend": "cdp", "lexmount_official_proxy": False},
    }


def record(
    task_id: str,
    *,
    verdict: str | None,
    status: str = "completed",
    infrastructure_failures: int = 0,
    policy_failures: int = 0,
    error_code: str | None = None,
) -> dict:
    events = []
    if error_code:
        events.append({"result": f"{error_code}: expected test failure"})
    result = {
        "task": {"task_id": task_id, "website": "example", "split": "smoke"},
        "status": status,
        "final_answer_status": "complete",
        "judge": {
            "status": "ok" if verdict else "skipped",
            "verdict": verdict,
            "reward": 1.0 if verdict == "yes" else (0.0 if verdict == "no" else None),
        },
        "guard": {
            "infrastructure_failures": infrastructure_failures,
            "policy_failures": policy_failures,
            "timeouts": 0,
            "termination_reason": "",
        },
        "events": events,
        "wall_seconds": 1.5,
    }
    if status != "completed":
        result["error"] = "browser setup failed"
    return result


def write_run(path: Path, *, backend: str, rows: list[dict], seed: int = 7) -> None:
    path.mkdir()
    (path / "run_manifest.json").write_text(
        json.dumps(manifest(backend=backend, seed=seed)), encoding="utf-8"
    )
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (path / "resource_summary.json").write_text(
        json.dumps(
            {
                "return_code": 0,
                "duration_seconds": 3.0,
                "sample_count": 3,
                "gpu_index": 0,
                "metrics": {"chrome_pss_gib": {"mean": 1.2}, "gpu_memory_mib_mean": 4000},
            }
        ),
        encoding="utf-8",
    )


def test_audit_separates_quality_and_infrastructure_denominators(tmp_path: Path) -> None:
    module = load_script_module()
    lexmount_dir = tmp_path / "lexmount"
    local_dir = tmp_path / "local"
    write_run(
        lexmount_dir,
        backend="lexmount",
        rows=[
            record("task-1", verdict="yes"),
            record(
                "task-2",
                verdict="no",
                infrastructure_failures=1,
                error_code="ERROR_INFRASTRUCTURE_NAVIGATE",
            ),
        ],
    )
    write_run(
        local_dir,
        backend="local",
        rows=[
            record("task-1", verdict="no"),
            record("task-2", verdict=None, status="setup_or_runner_error"),
        ],
    )

    audit = module.audit_pair(lexmount_dir, local_dir)

    assert audit["comparison_contract"] == {"matches": True, "differences": {}}
    assert audit["arms"]["lexmount"]["raw_success_rate"] == 0.5
    assert audit["arms"]["lexmount"]["quality_success_rate"] == 1.0
    assert audit["arms"]["local"]["raw_success_rate"] == 0.0
    assert audit["arms"]["local"]["quality_success_rate"] == 0.0
    assert audit["paired_quality"] == {
        "eligible_tasks": 1,
        "outcomes": {"lexmount_only_success": 1},
        "lexmount_successes": 1,
        "local_successes": 0,
    }
    assert audit["pairs"][1]["lexmount"]["event_error_codes"] == [
        "ERROR_INFRASTRUCTURE_NAVIGATE"
    ]
    assert audit["resources"]["lexmount"]["metrics"] == {
        "chrome_pss_gib": {"mean": 1.2},
        "gpu_memory_mib_mean": 4000,
    }


def test_audit_rejects_incomplete_task_coverage(tmp_path: Path) -> None:
    module = load_script_module()
    lexmount_dir = tmp_path / "lexmount"
    local_dir = tmp_path / "local"
    write_run(lexmount_dir, backend="lexmount", rows=[record("task-1", verdict="yes")])
    write_run(local_dir, backend="local", rows=[record("task-2", verdict="yes")])

    with pytest.raises(ValueError, match="paired task coverage differs"):
        module.audit_pair(lexmount_dir, local_dir)


def test_audit_marks_manifest_control_differences(tmp_path: Path) -> None:
    module = load_script_module()
    lexmount_dir = tmp_path / "lexmount"
    local_dir = tmp_path / "local"
    write_run(lexmount_dir, backend="lexmount", rows=[record("task-1", verdict="yes")])
    write_run(local_dir, backend="local", rows=[record("task-1", verdict="yes")], seed=8)

    audit = module.audit_pair(lexmount_dir, local_dir)

    assert audit["comparison_contract"]["matches"] is False
    assert set(audit["comparison_contract"]["differences"]) == {"generation"}


def test_audit_allows_expected_local_automation_difference(tmp_path: Path) -> None:
    module = load_script_module()
    lexmount_dir = tmp_path / "lexmount"
    local_dir = tmp_path / "local"
    write_run(lexmount_dir, backend="lexmount", rows=[record("task-1", verdict="yes")])
    write_run(local_dir, backend="local", rows=[record("task-1", verdict="yes")])
    lexmount_manifest = json.loads((lexmount_dir / "run_manifest.json").read_text())
    local_manifest = json.loads((local_dir / "run_manifest.json").read_text())
    lexmount_manifest["browser"]["local_disable_automation_controlled"] = False
    local_manifest["browser"]["local_disable_automation_controlled"] = True
    (lexmount_dir / "run_manifest.json").write_text(json.dumps(lexmount_manifest))
    (local_dir / "run_manifest.json").write_text(json.dumps(local_manifest))

    audit = module.audit_pair(lexmount_dir, local_dir)

    assert audit["comparison_contract"] == {
        "matches": True,
        "differences": {},
        "intentional_backend_differences": {
            "browser.local_disable_automation_controlled": {"lexmount": False, "local": True}
        },
    }
