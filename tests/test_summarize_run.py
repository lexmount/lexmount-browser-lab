from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_run import summarize_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summarize_run_keeps_planned_and_judged_denominators(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "website_region": "zh", "task_type": "T1"}),
                json.dumps({"id": 2, "website_region": "en", "task_type": "T2"}),
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    _write_json(run_dir / "config_snapshot.json", {"run": {"mode": "all"}})
    _write_json(
        run_dir / "tasks/1/result.json",
        {
            "metrics": {
                "steps": 3,
                "end_to_end_ms": 10_000,
                "usage": {"total_tokens": 100, "total_cost": 0.1},
            },
            "agent_done": "done",
            "agent_success": True,
            "env_status": "success",
            "error": None,
        },
    )
    _write_json(
        run_dir / "tasks_eval_result/task_eval_results.json",
        {"task_id": "1", "predicted_label": 1, "evaluation_details": {"score": 80}},
    )

    summary = summarize_run(run_dir, dataset)

    assert summary["counts"] == {
        "planned": 2,
        "trajectory": 1,
        "judged": 1,
        "success": 1,
        "agent_done": 1,
        "agent_success": 1,
    }
    assert summary["rates"]["success_per_planned"] == 0.5
    assert summary["rates"]["success_per_judged"] == 1.0
