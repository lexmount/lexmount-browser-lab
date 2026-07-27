"""Verl multi-turn adapter for the existing Lexmount CDP WebVoyager environment."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import socket
import threading
import time
from typing import Any

import aiohttp
from openai import AsyncOpenAI
from transformers.utils import get_json_schema
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from lexbrowser_webvoyager_no_anti_bot.environment import (
    LexmountDOMMode,
    TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT,
    TrajectoryGuard,
)

LOGGER = logging.getLogger(__name__)

_METRICS_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=4096)
_METRICS_THREAD: threading.Thread | None = None
_METRICS_THREAD_LOCK = threading.Lock()
JUDGE_TRANSCRIPT_CHAR_LIMIT = int(os.environ.get("LEXBROWSER_JUDGE_TRANSCRIPT_CHAR_LIMIT", "60000"))
TRUNCATION_MARKER = "...(truncated)..."


def _metrics_writer() -> None:
    output_dir = os.environ.get("LEXBROWSER_METRICS_DIR", "")
    if not output_dir:
        return
    path = os.path.join(output_dir, f"rollout-metrics-{socket.gethostname()}-{os.getpid()}.jsonl")
    stream = None
    while True:
        record = _METRICS_QUEUE.get()
        try:
            if stream is None:
                os.makedirs(output_dir, exist_ok=True)
                stream = open(path, "a", encoding="utf-8", buffering=1)
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
            stream = None
        finally:
            _METRICS_QUEUE.task_done()


def _emit_rollout_metrics(record: dict[str, Any]) -> None:
    """Best-effort, nonblocking metric emission; failures never reach training."""
    global _METRICS_THREAD
    if not os.environ.get("LEXBROWSER_METRICS_DIR"):
        return
    try:
        if _METRICS_THREAD is None:
            with _METRICS_THREAD_LOCK:
                if _METRICS_THREAD is None:
                    _METRICS_THREAD = threading.Thread(
                        target=_metrics_writer,
                        name="lexbrowser-metrics-writer",
                        daemon=True,
                    )
                    _METRICS_THREAD.start()
        _METRICS_QUEUE.put_nowait(record)
    except Exception:
        pass


def _add_metric(agent_data: AgentData, field: str, duration_s: float) -> None:
    try:
        value = max(0.0, float(duration_s))
        if isinstance(agent_data.metrics, dict):
            agent_data.metrics[field] = float(agent_data.metrics.get(field, 0.0)) + value
        else:
            current = float(getattr(agent_data.metrics, field, 0.0))
            setattr(agent_data.metrics, field, current + value)
    except Exception:
        pass


def _metric_value(agent_data: AgentData, field: str) -> float:
    try:
        if isinstance(agent_data.metrics, dict):
            return float(agent_data.metrics.get(field, 0.0))
        return float(getattr(agent_data.metrics, field, 0.0))
    except Exception:
        return 0.0


def _set_metric(agent_data: AgentData, field: str, value: float) -> None:
    """Best-effort assignment used by signed reconciliation metrics."""
    try:
        metric_value = float(value)
        if isinstance(agent_data.metrics, dict):
            agent_data.metrics[field] = metric_value
        else:
            setattr(agent_data.metrics, field, metric_value)
    except Exception:
        pass


def _truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = max(1, limit // 2)
    return text[:half] + TRUNCATION_MARKER + text[-half:], True


def _render_transcript(events: list[dict[str, str]]) -> tuple[str, bool]:
    if not events:
        return "", False
    call_lines = [
        f"TOOL_CALL browser({event['parameters']})"
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


def _extract_user_visible_answer(content: str, generation_truncated: bool) -> tuple[str, str]:
    """Never promote an unfinished Qwen reasoning block to the final answer."""
    text = content.strip()
    if generation_truncated:
        return "", "generation_truncated"
    if re.search(r"<think\b", text, flags=re.IGNORECASE):
        text = re.sub(
            r"<think\b[^>]*>.*?</think\s*>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if re.search(r"</?think\b", text, flags=re.IGNORECASE):
            return "", "no_final_answer"
    text = re.sub(r"^<final>\s*|\s*</final>$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"(?:<\|im_start\|>\s*assistant\s*|<\|im_end\|>|<\|endoftext\|>)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return (text, "complete") if text else ("", "no_final_answer")


class BrowserTool(BaseTool):
    """Keeps one CDP browser session per agent request and closes it at rollout end."""

    _sessions: dict[str, dict[str, Any]] = {}

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config, tool_schema)
        _BROWSER_TOOLS.append(self)
        self.service_url = str(
            config.get("nemo_gym_url") or os.environ.get("NEMO_GYM_BROWSER_URL", "")
        ).rstrip("/")
        self.mode: LexmountDOMMode | None = None
        if not self.service_url:
            self.mode = LexmountDOMMode(
                api_key=os.environ["LEXMOUNT_API_KEY"], project_id=os.environ["LEXMOUNT_PROJECT_ID"],
                dom_backend="cdp", stagehand_model="", policy_model="", proxy_model_to_stagehand=False,
                browser_mode="normal", official_proxy=False, external_proxy=None,
                max_concurrent_sessions=int(config.get("max_concurrent_sessions", 8)),
                session_create_timeout_s=60.0, stagehand_ready_timeout_s=30.0,
                setup_navigation_timeout_s=30.0, per_tool_timeout_s=25.0,
                episode_timeout_s=180.0, max_repeated_tool_calls=3,
            )

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as client:
            async with client.post(f"{self.service_url}{path}", json=payload) as response:
                body = await response.json()
                if response.status >= 400:
                    raise RuntimeError(f"NeMo-Gym browser service {path} failed: {body}")
                return body

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        def browser(operation: str, instruction: str = "") -> str:
            """Use the real browser.

            Args:
                operation: One of observe, act, or navigate.
                instruction: For act use a grounded selector action; for navigate use a URL.
            """
            return ""
        return OpenAIFunctionToolSchema(**get_json_schema(browser))

    async def _state(self, agent_data: AgentData) -> dict[str, Any]:
        request_id = agent_data.request_id
        if request_id in self._sessions:
            return self._sessions[request_id]
        create = agent_data.tools_kwargs["browser"]["create_kwargs"]
        if self.service_url:
            record = {
                "question": create["question"],
                "events": [],
                "tool_call_count": 0,
                "browser_error": "",
                "service": True,
                "tool": self,
            }
            started = time.perf_counter()
            try:
                try:
                    await self._post(
                        "/reset",
                        {
                            "session_id": request_id,
                            "question": create["question"],
                            "start_url": create["start_url"],
                            "task_id": create.get("task_id", ""),
                            "rubric": create.get("rubric", ""),
                        },
                        # The sidecar bounds the whole reset — creates, retries
                        # and initial navigation — by LEXMOUNT_RESET_TOTAL_BUDGET_S
                        # (110s default). This client deadline must stay above
                        # that budget so the server always finishes (or refuses)
                        # before the client walks away; a client-side timeout
                        # here would otherwise let the server register a session
                        # that no caller will ever step or close.
                        timeout=float(os.environ.get("LEXMOUNT_RESET_REQUEST_TIMEOUT_S", "120")),
                    )
                except Exception as exc:
                    record["reset_error"] = type(exc).__name__
            finally:
                _add_metric(agent_data, "browser_environment_init_s", time.perf_counter() - started)
            self._sessions[request_id] = record
            return record

        assert self.mode is not None
        started = time.perf_counter()
        state: dict[str, Any] = {"info": {"question": create["question"], "start_url": create["start_url"]}}
        reset_error = ""
        try:
            try:
                state = await self.mode.setup_state(state)
                session = state["browser_session"]
                session.set_task_query(create["question"])
                await self.mode.navigate(create["start_url"], session, state["trajectory_guard"])
            except Exception as exc:
                # Mirror the service path: a failed environment reset is a
                # classified rollout outcome, not a crashed agent loop.
                reset_error = type(exc).__name__
                await self.mode.cleanup_session(state)
        finally:
            _add_metric(agent_data, "browser_environment_init_s", time.perf_counter() - started)
        self._sessions[request_id] = {
            "state": state,
            "question": create["question"],
            "events": [],
            "tool_call_count": 0,
            "browser_error": "",
            "service": False,
            "tool": self,
        }
        if reset_error:
            self._sessions[request_id]["reset_error"] = reset_error
        return self._sessions[request_id]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data: AgentData = kwargs["agent_data"]
        record = await self._state(agent_data)
        operation = str(parameters.get("operation", ""))
        instruction = str(parameters.get("instruction", ""))
        record["tool_call_count"] += 1
        if record.get("reset_error"):
            result = f"ERROR_ENVIRONMENT_BROWSER_RESET: {record['reset_error']}"
            record["browser_error"] = result
        elif record["service"]:
            started = time.perf_counter()
            try:
                try:
                    payload = await self._post(
                        "/step",
                        {
                            "session_id": agent_data.request_id,
                            "operation": operation,
                            "instruction": instruction,
                        },
                        timeout=35.0,
                    )
                except Exception as exc:
                    record["browser_error"] = f"{type(exc).__name__}: {exc}"
                    raise
            finally:
                metric = "browser_observe_s" if operation == "observe" else "browser_act_s"
                _add_metric(agent_data, metric, time.perf_counter() - started)
            result = str(payload["observation"])
        else:
            state = record["state"]
            session, guard = state["browser_session"], state["trajectory_guard"]
            mode = record["tool"].mode
            assert mode is not None
            started = time.perf_counter()
            try:
                if operation == "observe":
                    result = await mode.observe(instruction, session, guard=guard)
                elif operation == "act":
                    result = await mode.act(instruction, session, guard=guard)
                elif operation == "navigate":
                    result = await mode.navigate(instruction, session, guard)
                else:
                    result = "ERROR_POLICY_TOOL: operation must be observe, act, or navigate"
            except Exception as exc:
                record["browser_error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                metric = "browser_observe_s" if operation == "observe" else "browser_act_s"
                _add_metric(agent_data, metric, time.perf_counter() - started)
        record["events"].append(
            {
                "parameters": json.dumps(parameters, ensure_ascii=False),
                "result": str(result),
            }
        )
        return ToolResponse(text=result), 0.0, {}

    @classmethod
    async def finalize(
        cls,
        agent_data: AgentData,
        final_answer: str,
        final_answer_status: str,
        generation_truncated: bool,
        episode_timed_out: bool,
    ) -> tuple[float, dict[str, Any]]:
        record = cls._sessions.pop(agent_data.request_id, None)
        if record is None:
            return 0.0, {
                "lexbrowser_reason": "no_browser_session",
                "lexbrowser_transcript": "",
                "environment_service": "nemo-gym",
            }
        transcript, transcript_truncated = _render_transcript(record["events"])
        execution_status = {
            "tool_call_count": int(record["tool_call_count"]),
            "session_created": not bool(record.get("reset_error")),
            "episode_timed_out": bool(episode_timed_out),
            "generation_truncated": bool(generation_truncated),
            "final_answer_present": bool(final_answer),
            "final_answer_status": final_answer_status,
            "transcript_truncated": transcript_truncated,
            "browser_error": str(record["browser_error"]),
        }
        if record["service"]:
            tool = record["tool"]
            result: dict[str, Any] | None = None
            close_error: Exception | None = None
            # /close is idempotent server-side (finished results are cached and
            # replayed), so a retry after a client timeout recovers the real
            # verdict instead of losing it to reward=0.
            for _ in range(2):
                try:
                    result = await tool._post(
                        "/close",
                        {
                            "session_id": agent_data.request_id,
                            "verify": True,
                            "final_answer": final_answer,
                            "final_answer_status": final_answer_status,
                            "generation_truncated": generation_truncated,
                            "episode_timed_out": episode_timed_out,
                            "final_answer_present": bool(final_answer),
                        },
                        timeout=90.0,
                    )
                    break
                except Exception as exc:
                    close_error = exc
            if result is None:
                assert close_error is not None
                return 0.0, {
                    "lexbrowser_reason": f"service_close_error:{type(close_error).__name__}",
                    "lexbrowser_transcript": transcript,
                    "environment_service": "nemo-gym",
                    "lexbrowser_invalid_sample": True,
                    "lexbrowser_invalid_reason": "environment_close_rpc_failed",
                }
            info = dict(result.get("info") or {})
            # Environment-failure classification (requested by the training
            # side): a rollout whose reward=0 comes from infrastructure — not
            # from the policy — must be distinguishable so the recipe can
            # excise it from the GRPO loss instead of learning from it.
            if record.get("reset_error"):
                info["lexbrowser_invalid_sample"] = True
                info["lexbrowser_invalid_reason"] = "environment_reset_failed"
            elif info.get("lexbrowser_reason") == "unknown_browser_session":
                info["lexbrowser_invalid_sample"] = True
                info["lexbrowser_invalid_reason"] = "environment_unknown_session"
            elif info.get("judge_status") == "error":
                # Previously raised judge_error_without_training_signal, which
                # aborted the whole training step; an unusable judge verdict is
                # an invalid sample, not a fatal condition.
                info["lexbrowser_invalid_sample"] = True
                info["lexbrowser_invalid_reason"] = "environment_judge_failed"
            return float(result["reward"]), info

        score, reason = 0.0, "no_tool_calls"
        session_cleanup_s = 0.0
        if record.get("reset_error"):
            mode = record["tool"].mode
            assert mode is not None
            await mode.cleanup_session(record.get("state") or {})
            return 0.0, {
                "lexbrowser_reason": "environment_reset_failed",
                "lexbrowser_transcript": transcript,
                "environment_service": "direct",
                "lexbrowser_invalid_sample": True,
                "lexbrowser_invalid_reason": "environment_reset_failed",
                **execution_status,
            }
        try:
            client = AsyncOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
                timeout=float(os.environ.get("LEXBROWSER_JUDGE_REQUEST_TIMEOUT_S", "45")),
            )
            prompt = TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT.format(
                question=record["question"],
                response=transcript,
                final_answer=final_answer,
                execution_status=json.dumps(execution_status, ensure_ascii=False, sort_keys=True),
                rubric="Not provided for this task.",
                initial_state=json.dumps(
                    {"start_url": record.get("state", {}).get("initial_url", "")},
                    ensure_ascii=False,
                ),
                final_url="Unavailable",
                final_state="Unavailable in direct mode.",
                screenshot_evidence="Unavailable: this run uses a text-only Judge.",
            )
            judge_started = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "glm-5.2"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=1024,
                )
            finally:
                judge_latency_s = time.perf_counter() - judge_started
            text = (response.choices[0].message.content or "").strip().lower()
            score, reason = (1.0, "judge_yes") if text.startswith("yes") else (0.0, f"judge:{text[:80]}")
        except Exception as exc:
            reason = f"judge_error:{type(exc).__name__}"
        finally:
            mode = record["tool"].mode
            assert mode is not None
            cleanup_started = time.perf_counter()
            try:
                await mode.cleanup_session(record["state"])
            finally:
                session_cleanup_s = time.perf_counter() - cleanup_started
        info = {
            "lexbrowser_reason": reason,
            "lexbrowser_transcript": transcript,
            "environment_service": "direct",
            "judge_latency_s": locals().get("judge_latency_s", 0.0),
            "judge_audit_write_s": 0.0,
            "browser_session_cleanup_s": session_cleanup_s,
            **execution_status,
        }
        if reason.startswith("judge_error"):
            info["lexbrowser_invalid_sample"] = True
            info["lexbrowser_invalid_reason"] = "environment_judge_failed"
        return score, info


_BROWSER_TOOLS: list[BaseTool] = []


@register("lexbrowser_tool_agent")
class LexBrowserToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        # Environment failures are resampled before they ever reach the batch:
        # a rollout invalidated by infrastructure (reset/close/judge transport)
        # is retried as a fresh episode with a new session.  Failed resets are
        # fast (bounded by the reset budget / circuit breaker), so a retry is
        # cheap exactly when it is needed.  Only a rollout that stays invalid
        # after the retries reaches the batch, loss-masked, where the patched
        # GRPO estimator excludes it from group statistics.
        max_attempts = 1 + max(0, int(os.environ.get("LEXBROWSER_ENV_FAILURE_RETRIES", "1")))
        output: AgentLoopOutput | None = None
        for attempt in range(1, max_attempts + 1):
            output = await self._run_episode(sampling_params, **kwargs)
            if attempt > 1:
                output.extra_fields["lexbrowser_env_retry_attempts"] = attempt
            if not output.extra_fields.get("lexbrowser_invalid_sample"):
                return output
            if attempt < max_attempts:
                LOGGER.warning(
                    "Environment-invalid rollout (%s); resampling episode (attempt %d/%d)",
                    output.extra_fields.get("lexbrowser_invalid_reason", "unknown"),
                    attempt + 1,
                    max_attempts,
                )
        assert output is not None
        return output

    async def _run_episode(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        rollout_started = time.perf_counter()
        action_max_tokens = int(os.environ.get("LEXBROWSER_ACTION_MAX_TOKENS", "1024"))
        if action_max_tokens < 1:
            raise ValueError("LEXBROWSER_ACTION_MAX_TOKENS must be positive")
        messages = list(kwargs["raw_prompt"])
        multimodal = await self.process_multi_modal_info(messages)
        agent_data = AgentData(messages, multimodal.get("images"), multimodal.get("videos"), multimodal.get("audios"), self._get_mm_processor_kwargs(multimodal.get("audios")), {}, __import__("uuid").uuid4().hex, kwargs.get("tools_kwargs", {}))
        agent_data._active_tools, agent_data._active_tool_schemas = self.tools, self.tool_schemas
        state = AgentState.PENDING
        episode_timeout_s = float(os.environ.get("LEXMOUNT_AGENT_EPISODE_TIMEOUT_S", "180"))
        timed_out = False
        last_generation_tokens = 0
        last_generation_truncated = False
        try:
            async with asyncio.timeout(episode_timeout_s):
                while state != AgentState.TERMINATED:
                    if state == AgentState.PENDING:
                        state = await self._handle_pending_state(agent_data, sampling_params)
                    elif state == AgentState.GENERATING:
                        turn_sampling_params = dict(sampling_params)
                        configured_max = turn_sampling_params.get(
                            "max_tokens", turn_sampling_params.get("max_new_tokens")
                        )
                        turn_sampling_params.pop("max_new_tokens", None)
                        turn_sampling_params["max_tokens"] = min(
                            int(configured_max) if configured_max is not None else action_max_tokens,
                            action_max_tokens,
                        )
                        generated_before = sum(agent_data.response_mask)
                        started = time.perf_counter()
                        try:
                            state = await self._handle_generating_state(
                                agent_data, turn_sampling_params
                            )
                            last_generation_tokens = sum(agent_data.response_mask) - generated_before
                            last_generation_truncated = (
                                last_generation_tokens >= int(turn_sampling_params["max_tokens"])
                            )
                        finally:
                            _add_metric(agent_data, "policy_generation_s", time.perf_counter() - started)
                    else:
                        state = await self._handle_processing_tools_state(agent_data)
        except TimeoutError:
            timed_out = True
            agent_data.extra_fields["lexbrowser_agent_timeout"] = episode_timeout_s
        final_answer = ""
        final_answer_status = "episode_timed_out" if timed_out else "no_final_answer"
        if not timed_out and agent_data.response_ids:
            active_tools = getattr(agent_data, "_active_tools", self.tools)
            tool_schemas = [tool.tool_schema for tool in active_tools.values()]
            try:
                assistant_content, final_tool_calls = await self.tool_parser.extract_tool_calls(
                    agent_data.response_ids, tool_schemas
                )
                if final_tool_calls:
                    final_answer_status = "tool_call_only"
                else:
                    final_answer, final_answer_status = _extract_user_visible_answer(
                        assistant_content or "", last_generation_truncated
                    )
            except Exception:
                final_answer = ""
                final_answer_status = "parse_error"
        finalize_started = time.perf_counter()
        try:
            score, info = await BrowserTool.finalize(
                agent_data,
                final_answer,
                final_answer_status,
                last_generation_truncated,
                timed_out,
            )
        finally:
            rollout_finalize_s = time.perf_counter() - finalize_started
            _add_metric(agent_data, "rollout_finalize_s", rollout_finalize_s)
        judge_latency_s = float(info.get("judge_latency_s") or 0.0)
        judge_audit_write_s = float(info.get("judge_audit_write_s") or 0.0)
        browser_session_cleanup_s = float(info.get("browser_session_cleanup_s") or 0.0)
        finalize_service_s = float(info.get("finalize_service_s") or rollout_finalize_s)
        finalize_rpc_overhead_s = rollout_finalize_s - finalize_service_s
        finalize_overhead_s = finalize_service_s - (
            judge_latency_s + judge_audit_write_s + browser_session_cleanup_s
        )
        _add_metric(agent_data, "judge_latency_s", judge_latency_s)
        _add_metric(agent_data, "judge_audit_write_s", judge_audit_write_s)
        _add_metric(agent_data, "browser_session_cleanup_s", browser_session_cleanup_s)
        _set_metric(agent_data, "rollout_finalize_service_s", finalize_service_s)
        _set_metric(agent_data, "rollout_finalize_rpc_overhead_s", finalize_rpc_overhead_s)
        _set_metric(agent_data, "rollout_finalize_overhead_s", finalize_overhead_s)
        if timed_out:
            info.setdefault("lexbrowser_reason", "agent_episode_timeout")
        info.setdefault("final_answer_status", final_answer_status)
        info.setdefault("generation_truncated", last_generation_truncated)
        info.setdefault("final_answer_present", bool(final_answer))
        response_length = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-response_length:] if response_length else []
        prompt_ids = agent_data.prompt_ids[:-response_length] if response_length else list(agent_data.prompt_ids)
        response_ids = response_ids[:self.response_length]
        response_mask = list(agent_data.response_mask[:self.response_length])
        response_logprobs = list(agent_data.response_logprobs[:self.response_length])
        empty_response = not response_ids
        if empty_response:
            # Keep every sampled trajectory in its GRPO group without creating a
            # fabricated policy gradient for a generation that never happened.
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                raise RuntimeError("Tokenizer has no eos_token_id for an empty agent response")
            response_ids = [int(eos_token_id)]
            response_mask = [0]
            # TransferQueue requires rollout_log_probs to exist consistently
            # across the batch. This token is loss-masked, so its placeholder
            # value cannot contribute a fabricated policy gradient.
            response_logprobs = [0.0]
        extra_fields = dict(agent_data.extra_fields)
        extra_fields.update(
            {
                "lexbrowser_final_answer_status": final_answer_status,
                "lexbrowser_generation_truncated": last_generation_truncated,
                "lexbrowser_last_generation_tokens": last_generation_tokens,
            }
        )
        if empty_response:
            extra_fields["lexbrowser_empty_response"] = True
        invalid_sample = bool(info.get("lexbrowser_invalid_sample"))
        invalid_reason = str(info.get("lexbrowser_invalid_reason") or "")
        if empty_response and not invalid_sample:
            # A zero-token episode has two causally different sources and only
            # one is an environment failure:
            # (a) infrastructure — the episode deadline expired (or generation
            #     was aborted) before the first generate() completed.  Exclude
            #     from the group baseline like any environment failure.
            # (b) policy — the model's first sampled token was a stop token
            #     and the engine strips it from the output.  That is a real
            #     policy outcome: reward=0 must participate in the baseline.
            #     The response stays loss-masked (the placeholder token and
            #     logprob are fabricated, so it must not carry a gradient).
            # Non-timeout aborts (e.g. engine preemption) currently land in
            # (b); verify the engine's stop-token behavior before tightening.
            if timed_out:
                invalid_sample = True
                invalid_reason = "generation_aborted_timeout"
            else:
                extra_fields["lexbrowser_policy_empty"] = True
        # Always materialize the flags on every sample: the batch collation
        # only exposes a non-tensor key when at least one sample carries it,
        # and the advantage patch must distinguish "no invalid samples"
        # (all-False) from "flags unavailable" (fall back to mask inference).
        extra_fields["lexbrowser_invalid_sample"] = invalid_sample
        extra_fields["lexbrowser_invalid_reason"] = invalid_reason
        if invalid_sample:
            # Environment failure (reset/close/judge infrastructure or an
            # aborted generation), not a policy failure.  Keep the trajectory
            # in its GRPO group but excise its policy gradient by loss-masking
            # every response token.  Group-statistics exclusion happens in the
            # patched advantage estimator via the explicit flag above.
            response_mask = [0] * len(response_mask)
        rollout_step = kwargs.get("global_steps", 0)
        if extra_fields.get("min_global_steps") is None:
            extra_fields["min_global_steps"] = int(rollout_step or 0)
        if extra_fields.get("max_global_steps") is None:
            extra_fields["max_global_steps"] = int(rollout_step or 0)
        extra_fields.update({"reward_extra_info": info, "turn_scores": [], "tool_rewards": []})
        rollout_e2e_s = time.perf_counter() - rollout_started
        _add_metric(agent_data, "rollout_e2e_s", rollout_e2e_s)
        top_level_accounted_s = sum(
            _metric_value(agent_data, field)
            for field in (
                "browser_environment_init_s",
                "browser_observe_s",
                "browser_act_s",
                "policy_generation_s",
                "rollout_finalize_s",
            )
        )
        _set_metric(
            agent_data,
            "rollout_unattributed_s",
            rollout_e2e_s - top_level_accounted_s,
        )
        _emit_rollout_metrics(
            {
                "schema_version": 2,
                "timestamp": time.time(),
                "step": int(rollout_step or 0),
                "rollout_id": agent_data.request_id,
                "reward": float(score),
                "invalid_sample": invalid_sample,
                "invalid_reason": invalid_reason,
                "browser_environment_init_s": _metric_value(
                    agent_data, "browser_environment_init_s"
                ),
                "browser_observe_s": _metric_value(agent_data, "browser_observe_s"),
                "browser_act_s": _metric_value(agent_data, "browser_act_s"),
                "browser_session_cleanup_s": _metric_value(agent_data, "browser_session_cleanup_s"),
                "policy_generation_s": _metric_value(agent_data, "policy_generation_s"),
                "judge_latency_s": _metric_value(agent_data, "judge_latency_s"),
                "judge_audit_write_s": _metric_value(agent_data, "judge_audit_write_s"),
                "rollout_finalize_s": _metric_value(agent_data, "rollout_finalize_s"),
                "rollout_finalize_service_s": _metric_value(agent_data, "rollout_finalize_service_s"),
                "rollout_finalize_rpc_overhead_s": _metric_value(agent_data, "rollout_finalize_rpc_overhead_s"),
                "rollout_finalize_overhead_s": _metric_value(agent_data, "rollout_finalize_overhead_s"),
                "rollout_unattributed_s": _metric_value(agent_data, "rollout_unattributed_s"),
                "rollout_e2e_s": _metric_value(agent_data, "rollout_e2e_s"),
            }
        )
        return AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask, response_logprobs=response_logprobs or None, reward_score=score, num_turns=agent_data.user_turns + agent_data.assistant_turns + 1, metrics=agent_data.metrics, extra_fields=extra_fields)
