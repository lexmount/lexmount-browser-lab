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


def test_select_common_available_writes_source_ordered_manifest(tmp_path: Path) -> None:
    module = load_script_module()
    tasks_path = tmp_path / "tasks.jsonl"
    task_rows = [
        {
            "task_id": "task-1",
            "question": "one",
            "start_url": "https://one.example.test",
            "website": "example",
        },
        {
            "task_id": "task-2",
            "question": "two",
            "start_url": "https://two.example.test",
            "website": "example",
        },
        {
            "task_id": "task-3",
            "question": "three",
            "start_url": "https://three.example.test",
            "website": "example",
        },
    ]
    tasks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in task_rows), encoding="utf-8"
    )

    def write_probe(path: Path, statuses: dict[str, str]) -> None:
        rows = [
            {"task": task, "status": statuses[task["task_id"]]}
            for task in reversed(task_rows)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    lexmount_probe = tmp_path / "lexmount.jsonl"
    local_probe = tmp_path / "local.jsonl"
    write_probe(
        lexmount_probe,
        {"task-1": "available", "task-2": "degraded_document", "task-3": "available"},
    )
    write_probe(
        local_probe,
        {"task-1": "available", "task-2": "available", "task-3": "available"},
    )
    output = tmp_path / "common.jsonl"

    result = module.select_common_available(
        module.build_parser().parse_args(
            [
                "select-common-available",
                "--tasks",
                str(tasks_path),
                "--lexmount-probe",
                str(lexmount_probe),
                "--local-probe",
                str(local_probe),
                "--output",
                str(output),
            ]
        )
    )

    assert result == 0
    assert [json.loads(line)["task_id"] for line in output.read_text().splitlines()] == [
        "task-1",
        "task-3",
    ]
    manifest = json.loads((tmp_path / "common.jsonl.selection.json").read_text())
    assert manifest["counts"] == {
        "source_tasks": 3,
        "probed_tasks": 3,
        "common_available": 2,
        "selected_tasks": 2,
    }
    assert manifest["probe_statuses"]["paired"] == {
        "available|available": 2,
        "degraded_document|available": 1,
    }


def test_select_common_available_rejects_different_probe_coverage(tmp_path: Path) -> None:
    module = load_script_module()
    tasks_path = tmp_path / "tasks.jsonl"
    first_task = {
        "task_id": "task-1",
        "question": "one",
        "start_url": "https://one.example.test",
        "website": "example",
    }
    second_task = {
        "task_id": "task-2",
        "question": "two",
        "start_url": "https://two.example.test",
        "website": "example",
    }
    tasks_path.write_text(
        json.dumps(first_task) + "\n" + json.dumps(second_task) + "\n", encoding="utf-8"
    )
    lexmount_probe = tmp_path / "lexmount.jsonl"
    local_probe = tmp_path / "local.jsonl"
    lexmount_probe.write_text(
        json.dumps({"task": first_task, "status": "available"}) + "\n", encoding="utf-8"
    )
    local_probe.write_text(
        json.dumps({"task": second_task, "status": "available"}) + "\n", encoding="utf-8"
    )

    args = module.build_parser().parse_args(
        [
            "select-common-available",
            "--tasks",
            str(tasks_path),
            "--lexmount-probe",
            str(lexmount_probe),
            "--local-probe",
            str(local_probe),
            "--output",
            str(tmp_path / "common.jsonl"),
        ]
    )

    with pytest.raises(ValueError, match="probe task coverage differs"):
        module.select_common_available(args)


def test_probe_parser_accepts_network_change_retries() -> None:
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
            "--network-change-retries",
            "2",
        ]
    )

    assert args.network_change_retries == 2


def test_parser_accepts_explicit_browser_context_overrides() -> None:
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
            "--context-locale",
            "en-US",
            "--context-timezone-id",
            "America/New_York",
            "--context-geolocation",
            "40.7128,-74.0060,50",
        ]
    )

    assert module.browser_context_overrides(args) == {
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "geolocation": {"latitude": 40.7128, "longitude": -74.006, "accuracy": 50.0},
    }


def test_parser_accepts_explicit_lexmount_external_proxy() -> None:
    module = load_script_module()

    args = module.build_parser().parse_args(
        [
            "probe",
            "--tasks",
            "tasks.jsonl",
            "--output-dir",
            "out",
            "--backend",
            "lexmount",
            "--lexmount-external-proxy-from-env",
        ]
    )

    assert args.lexmount_external_proxy_from_env is True


def test_external_proxy_requires_complete_lexmount_environment(monkeypatch) -> None:
    module = load_script_module()
    args = types.SimpleNamespace(
        backend="lexmount",
        lexmount_external_proxy_from_env=True,
        lexmount_official_proxy=False,
    )
    for name in (
        "LEXMOUNT_EXTERNAL_PROXY_SERVER",
        "LEXMOUNT_EXTERNAL_PROXY_USERNAME",
        "LEXMOUNT_EXTERNAL_PROXY_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="LEXMOUNT_EXTERNAL_PROXY_SERVER"):
        module.lexmount_external_proxy_from_env(args)


def test_external_proxy_is_explicit_and_does_not_use_official_proxy(monkeypatch) -> None:
    module = load_script_module()
    args = types.SimpleNamespace(
        backend="lexmount",
        lexmount_external_proxy_from_env=True,
        lexmount_official_proxy=False,
    )
    monkeypatch.setenv("LEXMOUNT_EXTERNAL_PROXY_SERVER", "https://proxy.example.test:8443")
    monkeypatch.setenv("LEXMOUNT_EXTERNAL_PROXY_USERNAME", "test-user")
    monkeypatch.setenv("LEXMOUNT_EXTERNAL_PROXY_PASSWORD", "test-password")

    assert module.lexmount_external_proxy_from_env(args) == {
        "type": "external",
        "server": "https://proxy.example.test:8443",
        "username": "test-user",
        "password": "test-password",
    }

    args.lexmount_official_proxy = True
    with pytest.raises(ValueError, match="cannot be combined"):
        module.lexmount_external_proxy_from_env(args)


def test_tool_error_metadata_distinguishes_policy_initiated_infrastructure() -> None:
    module = load_script_module()

    assert module.tool_error_metadata(
        "ERROR_INFRASTRUCTURE_NAVIGATE: infrastructure_anti_bot_challenge"
    ) == {
        "error_class": "infrastructure",
        "error_code": "ERROR_INFRASTRUCTURE_NAVIGATE",
    }
    assert module.tool_error_metadata("Navigated to https://example.test") == {}


def test_parser_rejects_invalid_context_geolocation() -> None:
    module = load_script_module()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "probe",
                "--tasks",
                "tasks.jsonl",
                "--output-dir",
                "out",
                "--backend",
                "local",
                "--context-geolocation",
                "200,0",
            ]
        )


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
    observed: dict[str, int | dict] = {
        "active": 0,
        "peak": 0,
        "mode_concurrency": 0,
        "context_overrides": {},
    }

    class FakeMode:
        def __init__(self, **kwargs) -> None:
            observed["mode_concurrency"] = kwargs["max_concurrent_sessions"]
            observed["context_overrides"] = kwargs["context_overrides"]

        async def teardown(self) -> None:
            return None

    async def fake_probe_task(*, task, mode, args):
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
    assert observed == {
        "active": 0,
        "peak": 2,
        "mode_concurrency": 2,
        "context_overrides": {},
    }
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["tasks"] == 4
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["browser"]["blocking_thread_workers"] == max(
        2, min(32, (os.cpu_count() or 1) + 4)
    )
