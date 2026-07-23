#!/usr/bin/env python3
"""NeMo-Gym resources service for WebVoyager browser backends."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient

from lexbrowser_webvoyager_no_anti_bot.environment import (
    LocalCDPMode,
    LexmountDOMMode,
    TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT,
)
from rollout_audit import append_judge_io, utc_timestamp


JUDGE_TRANSCRIPT_CHAR_LIMIT = int(os.environ.get("LEXBROWSER_JUDGE_TRANSCRIPT_CHAR_LIMIT", "60000"))
TRUNCATION_MARKER = "...(truncated)..."


def _truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    """Match Verl's middle truncation while reporting evidence loss."""
    if len(text) <= limit:
        return text, False
    if limit <= len(TRUNCATION_MARKER):
        return text[:limit], True
    content_budget = limit - len(TRUNCATION_MARKER)
    left = content_budget // 2
    right = content_budget - left
    return text[:left] + TRUNCATION_MARKER + text[-right:], True


def _render_transcript(events: list[dict[str, str]]) -> tuple[str, bool]:
    """Share the Judge budget across all turns so no interaction disappears."""
    if not events:
        return "", False
    call_lines = [
        f"TOOL_CALL browser(operation={event['operation']!r}, instruction={event['instruction']!r})"
        for event in events
    ]
    fixed_chars = sum(len(line) + len("\nTOOL_RESULT: \n") for line in call_lines)
    result_budget = max(512, (JUDGE_TRANSCRIPT_CHAR_LIMIT - fixed_chars) // len(events))
    rendered: list[str] = []
    truncated = False
    for line, event in zip(call_lines, events, strict=True):
        evidence, shortened = _truncate_middle(event["result"], result_budget)
        rendered.append(f"{line}\nTOOL_RESULT: {evidence}")
        truncated = truncated or shortened
    transcript, total_shortened = _truncate_middle("\n".join(rendered), JUDGE_TRANSCRIPT_CHAR_LIMIT)
    return transcript, truncated or total_shortened


def _extract_structured_judge_result(raw_text: str) -> dict[str, str] | None:
    """Accept only a non-empty reason and a binary verdict."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"reason", "verdict"}:
        return None
    if not isinstance(payload.get("verdict"), str) or not isinstance(payload.get("reason"), str):
        return None
    verdict = payload["verdict"].strip().lower()
    reason = payload["reason"].strip()
    if verdict not in {"yes", "no"} or not reason:
        return None
    return {"reason": reason, "verdict": verdict}


def _sanitize_final_answer(text: str) -> str:
    """Defense in depth: reasoning tags must never enter the Judge prompt."""
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if re.search(r"</?think\b", cleaned, flags=re.IGNORECASE):
        return ""
    return re.sub(
        r"(?:<\|im_start\|>\s*assistant\s*|<\|im_end\|>|<\|endoftext\|>)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()


def _initial_environment_state(record: dict[str, Any]) -> str:
    state = record.get("state", {})
    start_url = str(
        state.get("initial_url")
        or state.get("info", {}).get("start_url")
        or ""
    )
    for event in record.get("events", []):
        if event.get("operation") == "observe":
            return json.dumps(
                {"start_url": start_url, "first_observation": event.get("result", "")},
                ensure_ascii=False,
            )
    return json.dumps(
        {"start_url": start_url, "summary": "No initial observation was recorded."},
        ensure_ascii=False,
    )


async def _capture_final_browser_state(record: dict[str, Any]) -> tuple[str, str]:
    """Capture Judge evidence over the existing CDP session, without another LLM call."""
    session = record.get("state", {}).get("browser_session")
    if session is not None and callable(getattr(session, "observe", None)):
        try:
            snapshot = await asyncio.wait_for(asyncio.to_thread(session.observe), timeout=25.0)
            if isinstance(snapshot, dict):
                return (
                    str(snapshot.get("url") or ""),
                    json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
                )
        except Exception as exc:
            fallback = f"Final snapshot unavailable: {type(exc).__name__}"
        else:
            fallback = "Final snapshot unavailable: unsupported snapshot result"
    else:
        fallback = "Final snapshot unavailable for this browser backend"

    for event in reversed(record.get("events", [])):
        if event.get("operation") != "observe":
            continue
        try:
            snapshot = json.loads(str(event.get("result") or ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(snapshot, dict):
            return str(snapshot.get("url") or ""), json.dumps(
                snapshot, ensure_ascii=False, separators=(",", ":")
            )
    return "", fallback


class ResetRequest(BaseModel):
    session_id: str
    question: str
    start_url: str
    task_id: str = ""
    rubric: str = ""


class StepRequest(BaseModel):
    session_id: str
    operation: str
    instruction: str = ""


class CloseRequest(BaseModel):
    session_id: str
    verify: bool = True
    final_answer: str = ""
    final_answer_status: str = "no_final_answer"
    generation_truncated: bool = False
    episode_timed_out: bool = False
    final_answer_present: bool = False


class BrowserResponse(BaseModel):
    session_id: str
    observation: str = ""


class CloseResponse(BaseModel):
    session_id: str
    reward: float
    info: dict[str, Any] = Field(default_factory=dict)


class WebVoyagerResourcesServer(SimpleResourcesServer):
    """Stateful NeMo-Gym resource server; the policy remains owned by Verl."""

    _sessions: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _sessions_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _mode: LexmountDOMMode | LocalCDPMode = PrivateAttr()
    _browser_backend: str = PrivateAttr(default="lexmount")
    _max_concurrent_sessions: int = PrivateAttr(default=64)
    _max_concurrent_creates: int = PrivateAttr(default=16)
    _create_attempts: int = PrivateAttr(default=4)
    _create_slots: asyncio.Semaphore = PrivateAttr()
    _episode_timeout_s: float = PrivateAttr(default=180.0)
    _peak_active_sessions: int = PrivateAttr(default=0)
    _expired_sessions: int = PrivateAttr(default=0)
    _reset_requests: int = PrivateAttr(default=0)
    _reset_successes: int = PrivateAttr(default=0)
    _reset_failures: int = PrivateAttr(default=0)
    _step_requests: int = PrivateAttr(default=0)
    _step_failures: int = PrivateAttr(default=0)
    _judge_requests: int = PrivateAttr(default=0)
    _judge_successes: int = PrivateAttr(default=0)
    _judge_failures: int = PrivateAttr(default=0)
    _audit_writes: int = PrivateAttr(default=0)
    _audit_failures: int = PrivateAttr(default=0)

    def model_post_init(self, context: Any) -> None:
        del context
        self._max_concurrent_sessions = int(os.environ.get("LEXMOUNT_MAX_CONCURRENT_SESSIONS", "64"))
        self._max_concurrent_creates = int(os.environ.get("LEXMOUNT_MAX_CONCURRENT_CREATES", "16"))
        # One initial create plus three retries; every create has its own 60s timeout.
        self._create_attempts = int(os.environ.get("LEXMOUNT_SESSION_CREATE_ATTEMPTS", "4"))
        self._episode_timeout_s = float(os.environ.get("LEXMOUNT_AGENT_EPISODE_TIMEOUT_S", "180"))
        self._create_slots = asyncio.Semaphore(self._max_concurrent_creates)
        self._browser_backend = os.environ.get("BROWSER_BACKEND", "lexmount").strip().lower()
        common_mode_kwargs = {
            "max_concurrent_sessions": self._max_concurrent_sessions,
            "setup_navigation_timeout_s": 30.0,
            "per_tool_timeout_s": 25.0,
            "episode_timeout_s": 180.0,
            "max_repeated_tool_calls": 3,
        }
        if self._browser_backend == "lexmount":
            self._mode = LexmountDOMMode(
                api_key=os.environ["LEXMOUNT_API_KEY"],
                project_id=os.environ["LEXMOUNT_PROJECT_ID"],
                dom_backend="cdp",
                stagehand_model="",
                policy_model="",
                proxy_model_to_stagehand=False,
                browser_mode=os.environ.get("LEXMOUNT_BROWSER_MODE", "normal"),
                official_proxy=os.environ.get("LEXMOUNT_OFFICIAL_PROXY", "0") == "1",
                external_proxy=None,
                session_create_timeout_s=60.0,
                stagehand_ready_timeout_s=30.0,
                **common_mode_kwargs,
            )
        elif self._browser_backend == "local_cdp":
            self._mode = LocalCDPMode(
                cdp_http_url=os.environ.get(
                    "LOCAL_CDP_HTTP_URL", "http://127.0.0.1:9222"
                ),
                **common_mode_kwargs,
            )
        else:
            raise ValueError(
                "BROWSER_BACKEND must be either 'lexmount' or 'local_cdp'; "
                f"got {self._browser_backend!r}"
            )

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI):
            try:
                yield
            finally:
                await self._cleanup_all()
                await self._mode.teardown()

        app = FastAPI(title="NeMo-Gym WebVoyager Resources", lifespan=lifespan)
        app.get("/health")(self.health)
        app.post("/reset", response_model=BrowserResponse)(self.reset)
        app.post("/step", response_model=BrowserResponse)(self.step)
        app.post("/verify", response_model=CloseResponse)(self.verify)
        app.post("/close", response_model=CloseResponse)(self.close)
        return app

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "framework": "nemo-gym",
            "backend": self._browser_backend,
            "active_sessions": len(self._sessions),
            "peak_active_sessions": self._peak_active_sessions,
            "max_concurrent_sessions": self._max_concurrent_sessions,
            "max_concurrent_creates": self._max_concurrent_creates,
            "session_create_timeout_s": 60.0,
            "session_create_max_attempts": self._create_attempts,
            "session_create_retries": max(0, self._create_attempts - 1),
            "episode_timeout_s": self._episode_timeout_s,
            "expired_sessions": self._expired_sessions,
            "pending_late_session_cleanups": self._mode.pending_late_session_cleanups,
            "reset_requests": self._reset_requests,
            "reset_successes": self._reset_successes,
            "reset_failures": self._reset_failures,
            "step_requests": self._step_requests,
            "step_failures": self._step_failures,
            "judge_requests": self._judge_requests,
            "judge_successes": self._judge_successes,
            "judge_failures": self._judge_failures,
            "audit_writes": self._audit_writes,
            "audit_failures": self._audit_failures,
            "audit_enabled": bool(os.environ.get("LEXBROWSER_AUDIT_DIR")),
        }

    async def reset(self, body: ResetRequest) -> BrowserResponse:
        self._reset_requests += 1
        await self._discard(body.session_id)
        state: dict[str, Any] = {}
        last_error: Exception | None = None
        reset_succeeded = False
        for attempt in range(1, self._create_attempts + 1):
            state = {"info": {"question": body.question, "start_url": body.start_url}}
            try:
                async with self._create_slots:
                    state = await self._mode.setup_state(state)
                session = state["browser_session"]
                session.set_task_query(body.question)
                await self._mode.navigate(body.start_url, session, state["trajectory_guard"])
                reset_succeeded = True
                break
            except Exception as exc:
                last_error = exc
                await self._mode.cleanup_session(state)
        if not reset_succeeded:
            self._reset_failures += 1
            assert last_error is not None
            raise last_error

        async with self._sessions_lock:
            record = {
                "state": state,
                "question": body.question,
                "task_id": body.task_id,
                "rubric": body.rubric,
                "events": [],
                "tool_call_count": 0,
                "browser_error": "",
                "lock": asyncio.Lock(),
            }
            self._sessions[body.session_id] = record
            record["expiry_task"] = asyncio.create_task(
                self._expire_session(body.session_id, record)
            )
            self._peak_active_sessions = max(self._peak_active_sessions, len(self._sessions))
            self._reset_successes += 1
        return BrowserResponse(session_id=body.session_id)

    async def step(self, body: StepRequest) -> BrowserResponse:
        self._step_requests += 1
        record = self._sessions.get(body.session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown browser session")
        async with record["lock"]:
            record["tool_call_count"] += 1
            state = record["state"]
            session = state["browser_session"]
            guard = state["trajectory_guard"]
            try:
                if body.operation == "observe":
                    result = await self._mode.observe(body.instruction, session, guard=guard)
                elif body.operation == "act":
                    result = await self._mode.act(body.instruction, session, guard=guard)
                elif body.operation == "navigate":
                    result = await self._mode.navigate(body.instruction, session, guard)
                else:
                    result = "ERROR_POLICY_TOOL: operation must be observe, act, or navigate"
            except Exception as exc:
                self._step_failures += 1
                record["browser_error"] = f"{type(exc).__name__}: {exc}"
                raise
            record["events"].append(
                {"operation": body.operation, "instruction": body.instruction, "result": str(result)}
            )
        return BrowserResponse(session_id=body.session_id, observation=result)

    async def verify(self, body: CloseRequest) -> CloseResponse:
        body.verify = True
        return await self.close(body)

    async def close(self, body: CloseRequest) -> CloseResponse:
        finalize_started = time.perf_counter()
        async with self._sessions_lock:
            record = self._sessions.pop(body.session_id, None)
        if record is None:
            return CloseResponse(
                session_id=body.session_id,
                reward=0.0,
                info={
                    "lexbrowser_reason": "unknown_browser_session",
                    "lexbrowser_transcript": "",
                    "environment_service": "nemo-gym",
                    "judge_audit_write_s": 0.0,
                    "browser_session_cleanup_s": 0.0,
                    "finalize_service_s": time.perf_counter() - finalize_started,
                },
            )

        reward, reason = 0.0, "verification_skipped"
        judge_audit: dict[str, Any] = {"status": "skipped"}
        self._cancel_expiry_task(record)
        transcript, transcript_truncated = _render_transcript(record["events"])
        final_answer = _sanitize_final_answer(body.final_answer)
        final_answer_status = body.final_answer_status if final_answer else "no_final_answer"
        final_url, final_state = await _capture_final_browser_state(record)
        initial_state = _initial_environment_state(record)
        execution_status = {
            "tool_call_count": int(record["tool_call_count"]),
            "session_created": True,
            "episode_timed_out": bool(body.episode_timed_out),
            "generation_truncated": bool(body.generation_truncated),
            "final_answer_present": bool(final_answer),
            "final_answer_status": str(final_answer_status),
            "transcript_truncated": transcript_truncated,
            "browser_error": str(record["browser_error"]),
        }
        judge_log_path = ""
        audit_write_s = 0.0
        session_cleanup_s = 0.0
        try:
            if body.verify:
                reward, reason, judge_audit = await self._judge(
                    record["question"],
                    transcript,
                    final_answer,
                    execution_status,
                    initial_state,
                    final_url,
                    final_state,
                    record.get("rubric", ""),
                )
            judge_output = judge_audit.get("raw_response")
            if not judge_output:
                judge_output = judge_audit.get("error_message") or judge_audit.get("reason", "")
            audit_started = time.perf_counter()
            try:
                try:
                    judge_log_path = await asyncio.to_thread(
                        append_judge_io,
                        {
                            "rollout_id": body.session_id,
                            "question": record["question"],
                            "judge_model": judge_audit.get("model"),
                            "judge_status": judge_audit.get("status"),
                            "judge_input": judge_audit.get("prompt", ""),
                            "judge_output": judge_output,
                            "judge_reasoning_content": judge_audit.get("reasoning_content", ""),
                            "judge_verdict": judge_audit.get("verdict"),
                            "judge_reason": judge_audit.get("verdict_reason"),
                            "reward": reward,
                            "reason": judge_audit.get("verdict_reason") or reason,
                            "duration_seconds": judge_audit.get("duration_seconds"),
                        },
                    )
                    if judge_log_path:
                        self._audit_writes += 1
                except Exception:
                    self._audit_failures += 1
            finally:
                audit_write_s = time.perf_counter() - audit_started
        finally:
            cleanup_started = time.perf_counter()
            try:
                await self._mode.cleanup_session(record["state"])
            finally:
                session_cleanup_s = time.perf_counter() - cleanup_started
        finalize_service_s = time.perf_counter() - finalize_started
        return CloseResponse(
            session_id=body.session_id,
            reward=reward,
            info={
                "lexbrowser_reason": reason,
                "lexbrowser_transcript": transcript,
                "environment_service": "nemo-gym",
                "judge_status": judge_audit.get("status", "unknown"),
                "judge_verdict": judge_audit.get("verdict"),
                "judge_reason": judge_audit.get("verdict_reason", ""),
                "judge_log_file": judge_log_path,
                "judge_latency_s": float(judge_audit.get("duration_seconds") or 0.0),
                "judge_audit_write_s": audit_write_s,
                "browser_session_cleanup_s": session_cleanup_s,
                "finalize_service_s": finalize_service_s,
                **execution_status,
            },
        )

    async def _judge(
        self,
        question: str,
        transcript: str,
        final_answer: str,
        execution_status: dict[str, Any],
        initial_state: str,
        final_url: str,
        final_state: str,
        rubric: str,
    ) -> tuple[float, str, dict[str, Any]]:
        if not transcript:
            return 0.0, "no_tool_calls", {"status": "skipped", "reason": "no_tool_calls"}
        started = time.monotonic()
        model = os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "glm-5.2")
        max_attempts = max(1, int(os.environ.get("LEXBROWSER_JUDGE_MAX_ATTEMPTS", "3")))
        prompt = TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT.format(
            question=question,
            response=transcript,
            final_answer=final_answer.strip(),
            execution_status=json.dumps(execution_status, ensure_ascii=False, sort_keys=True),
            rubric=rubric.strip() or "Not provided for this task.",
            initial_state=initial_state,
            final_url=final_url or "Unavailable",
            final_state=final_state,
            screenshot_evidence="Unavailable: this run uses a text-only Judge.",
        )
        audit: dict[str, Any] = {
            "status": "started",
            "started_at": utc_timestamp(),
            "model": model,
            "prompt": prompt,
        }
        client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
        attempts: list[dict[str, Any]] = []
        last_raw_text = ""
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            self._judge_requests += 1
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                message = response.choices[0].message
                last_raw_text = message.content or ""
                reasoning_content = getattr(message, "reasoning_content", "") or ""
                result = _extract_structured_judge_result(last_raw_text)
                if result is None and not last_raw_text.strip():
                    result = _extract_structured_judge_result(reasoning_content)
                attempt_record = {
                    "attempt": attempt,
                    "response_id": getattr(response, "id", ""),
                    "raw_response": last_raw_text,
                    "reasoning_content": reasoning_content,
                    "parsed_result": result,
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                }
                attempts.append(attempt_record)
                if result is not None:
                    self._judge_successes += 1
                    audit.update(
                        {
                            "status": "ok",
                            "duration_seconds": time.monotonic() - started,
                            "attempt_count": attempt,
                            "attempts": attempts,
                            "verdict": result["verdict"],
                            "verdict_reason": result["reason"],
                            **attempt_record,
                        }
                    )
                    return (
                        (1.0, "judge_yes", audit)
                        if result["verdict"] == "yes"
                        else (0.0, "judge_no", audit)
                    )
                last_error = "Judge returned empty or invalid reason/verdict JSON"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {"attempt": attempt, "error_type": type(exc).__name__, "error_message": str(exc)}
                )

        self._judge_failures += 1
        audit.update(
            {
                "status": "error",
                "duration_seconds": time.monotonic() - started,
                "attempt_count": max_attempts,
                "attempts": attempts,
                "raw_response": last_raw_text,
                "parsed_result": None,
                "error_type": "InvalidJudgeResponse",
                "error_message": last_error or "Judge produced no valid reason/verdict JSON",
            }
        )
        return 0.0, "judge_error:InvalidJudgeResponse", audit

    async def _discard(self, session_id: str) -> None:
        async with self._sessions_lock:
            record = self._sessions.pop(session_id, None)
        if record is not None:
            self._cancel_expiry_task(record)
            await self._mode.cleanup_session(record["state"])

    async def _expire_session(self, session_id: str, record: dict[str, Any]) -> None:
        try:
            await asyncio.sleep(self._episode_timeout_s)
            async with self._sessions_lock:
                if self._sessions.get(session_id) is not record:
                    return
                self._sessions.pop(session_id, None)
            self._expired_sessions += 1
            await self._mode.cleanup_session(record["state"])
        except asyncio.CancelledError:
            return

    @staticmethod
    def _cancel_expiry_task(record: dict[str, Any]) -> None:
        task = record.get("expiry_task")
        if isinstance(task, asyncio.Task) and task is not asyncio.current_task():
            task.cancel()

    async def _cleanup_all(self) -> None:
        async with self._sessions_lock:
            records = list(self._sessions.values())
            self._sessions.clear()
        for record in records:
            self._cancel_expiry_task(record)
        await asyncio.gather(
            *(self._mode.cleanup_session(record["state"]) for record in records),
            return_exceptions=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18180)
    args = parser.parse_args()

    config = BaseResourcesServerConfig(
        name="lexbrowser_webvoyager",
        host=args.host,
        port=args.port,
        entrypoint=__file__,
        domain="agent",
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host=args.host, port=args.port),
        global_config_dict=OmegaConf.create({}),
    )
    server = WebVoyagerResourcesServer(config=config, server_client=client)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
