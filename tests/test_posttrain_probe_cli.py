from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

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


def test_browser_setup_error_preserves_timeout_type() -> None:
    module = load_script_module()

    class FailingMode:
        async def setup_state(self, state):
            raise TimeoutError

        async def cleanup_session(self, state):
            return None

    args = types.SimpleNamespace(setup_attempts=1)
    task = module.Task("task", "question", "https://example.test", "example")

    with pytest.raises(RuntimeError, match="TimeoutError"):
        asyncio.run(module._open_browser_state(FailingMode(), task, args))


def test_browser_setup_uses_dedicated_navigation_timeout() -> None:
    module = load_script_module()
    observed: dict[str, float] = {}

    class SetupMode:
        async def setup_state(self, state):
            state["browser_session"] = object()
            state["trajectory_guard"] = object()
            return state

        async def navigate(self, url, session, guard, *, timeout_s):
            observed["timeout_s"] = timeout_s
            return f"Navigated to {url}"

        async def cleanup_session(self, state):
            return None

    args = types.SimpleNamespace(setup_attempts=1, setup_navigation_timeout=60.0)
    task = module.Task("task", "question", "https://example.test", "example")

    _, attempts = asyncio.run(module._open_browser_state(SetupMode(), task, args))

    assert attempts == 1
    assert observed == {"timeout_s": 60.0}


def test_judge_omits_temperature_when_not_requested() -> None:
    module = load_script_module()
    requests: list[dict] = []

    class Completions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            message = types.SimpleNamespace(content='{"verdict":"yes","reason":"evidence"}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    task = module.Task("task", "question", "https://example.test", "example")

    result = asyncio.run(
        module._judge_task(
            client,
            model="gpt-5.5",
            temperature=None,
            task=task,
            transcript="browser evidence",
            final_answer="answer",
            execution_status={},
            final_url="https://example.test",
            final_state="{}",
        )
    )

    assert result["verdict"] == "yes"
    assert requests[0]["model"] == "gpt-5.5"
    assert "temperature" not in requests[0]


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
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["browser"]["blocking_thread_workers"] == max(
        2, min(32, (os.cpu_count() or 1) + 4)
    )
