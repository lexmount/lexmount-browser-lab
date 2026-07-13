from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.summarize_replays import (
    load_json_object,
    load_json_records,
    summarize_replays,
    synthetic_evaluation_ids,
)


def _summary(run_id: str, first: bool, second: bool) -> dict:
    return {
        "run_dir": f"/tmp/{run_id}",
        "per_task": {
            "1": {"success": first, "judge_score": 80, "agent_done": "done", "signals": {}},
            "2": {"success": second, "judge_score": 40, "agent_done": "done", "signals": {}},
        },
    }


def test_summarize_replays_counts_repeated_successes() -> None:
    result = summarize_replays(
        [_summary("lex-1", True, False), _summary("lex-2", True, True)],
        [_summary("local-1", False, False), _summary("local-2", True, False)],
    )

    assert result["aggregate"]["lexmount"]["successes"] == 3
    assert result["aggregate"]["local"]["successes"] == 1
    assert result["tasks"][0]["lexmount"]["successes"] == 2


def test_summarize_replays_rejects_empty_arm() -> None:
    with pytest.raises(ValueError, match="at least one summary"):
        summarize_replays([], [_summary("local", True, True)])


def test_summarize_replays_rejects_disjoint_task_sets() -> None:
    local = _summary("local", True, True)
    local["per_task"] = {"3": local["per_task"].pop("2")}

    with pytest.raises(ValueError, match="do not share any task ids"):
        summarize_replays([_summary("lex", True, True)], [local])


def test_load_json_object_reports_source_path(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary\.json: invalid JSON object"):
        load_json_object(path)


def test_summarize_replays_requires_per_task_object() -> None:
    malformed = {"run_dir": "/tmp/run", "per_task": []}

    with pytest.raises(ValueError, match="valid 'per_task' object"):
        summarize_replays([malformed], [_summary("local", True, True)])


def test_load_json_records_reports_malformed_line(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text('{"task_id": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"records\.json:2: invalid JSON record"):
        load_json_records(path)


def test_synthetic_evaluation_requires_task_id(tmp_path) -> None:
    eval_dir = tmp_path / "tasks_eval_result"
    eval_dir.mkdir()
    path = eval_dir / "task_eval_results.json"
    path.write_text(
        '{"evaluation_details":{"benchmark_details":{"is_synthetic_failure":true}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic evaluation is missing task_id"):
        synthetic_evaluation_ids({"run_dir": str(tmp_path)})
