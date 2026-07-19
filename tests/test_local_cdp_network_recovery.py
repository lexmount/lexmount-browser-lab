from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training" / "lexbrowser_webvoyager" / "src"
sys.path.insert(0, str(SOURCE))

pytest.importorskip("verifiers")

from lexbrowser_webvoyager_no_anti_bot import environment


class FakeWebSocket:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.sent: list[dict[str, object]] = []

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def recv(self) -> str:
        return json.dumps(self.responses.pop(0))


def test_cdp_call_records_network_loading_failure() -> None:
    session = object.__new__(environment.LexmountCDPSession)
    session._next_id = 0
    session._session_id = "session-1"
    session._navigation_network_errors = []
    session._ws = FakeWebSocket(
        [
            {
                "method": "Network.loadingFailed",
                "sessionId": "session-1",
                "params": {"errorText": "net::ERR_NETWORK_CHANGED"},
            },
            {"id": 1, "result": {}},
        ]
    )

    assert session.call("Runtime.evaluate") == {}
    assert session.has_transient_network_change() is True


def test_cdp_navigation_reuses_session_after_network_change(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecoveringSession(environment.LexmountCDPSession):
        def __init__(self) -> None:
            self.navigation_calls = 0
            self._navigation_network_errors: list[str] = []

        def navigate(self, url: str) -> None:
            self.navigation_calls += 1
            self._navigation_network_errors.clear()

        def wait_for_usable_document(self, timeout_s: float) -> None:
            if self.navigation_calls == 1:
                self._navigation_network_errors.append("net::ERR_NETWORK_CHANGED")
                raise RuntimeError("infrastructure_browser_error_page: ERR_NETWORK_CHANGED")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(environment.asyncio, "sleep", no_sleep)
    mode = environment.LexmountDOMMode(
        api_key=None,
        project_id=None,
        browser_backend="local",
        dom_backend="cdp",
        stagehand_model="",
        policy_model="",
        proxy_model_to_stagehand=False,
        browser_mode="normal",
        official_proxy=False,
        external_proxy=None,
        local_chrome_executable_path=None,
        local_chrome_headless=True,
        local_proxy_server=None,
        local_proxy_bypass=None,
        network_change_retries=1,
        max_concurrent_sessions=1,
        session_create_timeout_s=1.0,
        stagehand_ready_timeout_s=1.0,
        setup_navigation_timeout_s=1.0,
        per_tool_timeout_s=1.0,
        episode_timeout_s=1.0,
        max_repeated_tool_calls=3,
    )
    session = RecoveringSession()

    retries = asyncio.run(
        mode._navigate_cdp_with_network_change_recovery(
            session, "https://example.test", timeout_s=1.0
        )
    )

    assert retries == 1
    assert session.navigation_calls == 2


def test_lexmount_cleanup_caps_high_concurrency_fanout() -> None:
    mode = environment.LexmountDOMMode(
        api_key=None,
        project_id=None,
        browser_backend="local",
        dom_backend="cdp",
        stagehand_model="",
        policy_model="",
        proxy_model_to_stagehand=False,
        browser_mode="normal",
        official_proxy=False,
        external_proxy=None,
        local_chrome_executable_path=None,
        local_chrome_headless=True,
        local_proxy_server=None,
        local_proxy_bypass=None,
        network_change_retries=1,
        max_concurrent_sessions=64,
        session_create_timeout_s=1.0,
        stagehand_ready_timeout_s=1.0,
        setup_navigation_timeout_s=1.0,
        per_tool_timeout_s=1.0,
        episode_timeout_s=1.0,
        max_repeated_tool_calls=3,
    )
    lock = threading.Lock()
    active = 0
    peak = 0

    class SlowSessions:
        def delete(self, *, session_id: str) -> None:
            nonlocal active, peak
            assert session_id.startswith("session-")
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1

    mode.lexmount = types.SimpleNamespace(sessions=SlowSessions())

    async def close_all() -> None:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=64))
        await asyncio.gather(
            *(
                mode._close_lexmount_session(
                    types.SimpleNamespace(id=f"session-{index}"), reason="test"
                )
                for index in range(64)
            )
        )

    asyncio.run(close_all())

    assert active == 0
    assert peak == 8
