from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "scripts" / "webvoyager_posttrain_eval.py"


def load_script_module() -> types.ModuleType:
    module_name = "webvoyager_posttrain_eval_for_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_parser_accepts_explicit_concurrency() -> None:
    module = load_script_module()

    args = module.build_parser().parse_args(
        [
            "probe",
            "--tasks",
            "tasks.jsonl",
            "--output-dir",
            "out",
            "--backend",
            "local",
            "--concurrency",
            "64",
        ]
    )

    assert args.concurrency == 64


def test_probe_uses_requested_session_concurrency(tmp_path, monkeypatch) -> None:
    module = load_script_module()
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"task-{index}",
                    "question": "test task",
                    "start_url": "https://example.test",
                    "website": "example",
                }
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    observed: dict[str, int] = {"active": 0, "peak": 0, "mode_concurrency": 0}

    class FakeMode:
        def __init__(self, **kwargs) -> None:
            observed["mode_concurrency"] = kwargs["max_concurrent_sessions"]
            self.slots = asyncio.Semaphore(kwargs["max_concurrent_sessions"])

        async def teardown(self) -> None:
            return None

    async def fake_probe_task(*, task, mode, args):
        async with mode.slots:
            observed["active"] += 1
            observed["peak"] = max(observed["peak"], observed["active"])
            try:
                await asyncio.sleep(0.01)
                return {
                    "task": task.as_dict(),
                    "backend": args.backend,
                    "status": "available",
                    "document": {"visible_text_chars": 160, "element_count": 2},
                    "setup": {"attempts": 1},
                    "wall_seconds": 0.01,
                }
            finally:
                observed["active"] -= 1

    package = types.ModuleType("lexbrowser_webvoyager_no_anti_bot")
    environment = types.ModuleType("lexbrowser_webvoyager_no_anti_bot.environment")
    environment.LexmountDOMMode = FakeMode
    monkeypatch.setitem(sys.modules, "lexbrowser_webvoyager_no_anti_bot", package)
    monkeypatch.setitem(sys.modules, "lexbrowser_webvoyager_no_anti_bot.environment", environment)
    monkeypatch.setattr(module, "probe_task", fake_probe_task)
    monkeypatch.setattr(module, "repository_revision", lambda: "test-revision")

    output_dir = tmp_path / "out"
    args = module.build_parser().parse_args(
        [
            "probe",
            "--tasks",
            str(tasks_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "local",
            "--concurrency",
            "2",
        ]
    )

    assert asyncio.run(module.run_probe(args)) == 0
    assert observed == {"active": 0, "peak": 2, "mode_concurrency": 2}
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["tasks"] == 4
