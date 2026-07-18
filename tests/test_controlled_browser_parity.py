from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "controlled-browser-parity" / "generate_tasks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("controlled_browser_parity_tasks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_formula_matches_documented_examples() -> None:
    module = load_module()

    assert module.record_for(1) == ("PRD-0001", "$17.13", "15 units")
    assert module.record_for(50) == ("PRD-0050", "$93.50", "24 units")


def test_task_manifest_has_exact_rubric_and_public_base_url(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "tasks.jsonl"

    assert (
        module.main(
            [
                "--base-url",
                "https://fixture.example",
                "--output",
                str(output),
                "--count",
                "2",
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in rows] == ["catalog-001", "catalog-002"]
    assert {row["start_url"] for row in rows} == {"https://fixture.example/"}
    assert rows[0]["expected_answer"] == {
        "must_include": ["$17.13", "15 units"],
        "minimum_act_events": 2,
    }
