# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Compatibility copy of NeMo Gym v0.3.0rc0's agent with tool-call preservation.
# ruff: noqa: B008, B010, B905, E501, I001, UP037
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import logging
import traceback
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import verifiers as vf
import verifiers.envs.multiturn_env as _multiturn_env_module
from fastapi import Body, Request, Response
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import ConfigDict, Field
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.response_utils import parse_response_messages as _original_parse_response_messages

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_global_aiohttp_client


logger = logging.getLogger(__name__)

# A rollout is handled by several asyncio tasks (browser environment, vLLM
# client, and judge).  Keep a request-local, secret-free timing record so the
# audit can distinguish model work from browser/provider waiting.  This is
# entirely inside the Gym agent app; it neither patches NeMo-RL nor changes
# sampling, rewards, or token reconstruction.
_ACTIVE_ROLLOUT_TIMING: ContextVar[dict[str, Any] | None] = ContextVar(
    "active_lexbrowser_rollout_timing", default=None
)
_SHARED_SEMAPHORES: dict[tuple[int, str, int], asyncio.Semaphore] = {}


async def _shared_semaphore(kind: str, limit: int) -> Any:
    """Return a process-wide Gym semaphore, rather than one per HTTP request."""
    if limit < 0:
        return await maybe_semaphore(limit)
    loop = asyncio.get_running_loop()
    key = (id(loop), kind, limit)
    semaphore = _SHARED_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _SHARED_SEMAPHORES[key] = semaphore
    return semaphore


def _record_timing_event(
    timing: dict[str, Any] | None,
    metric: str,
    started_at_monotonic: float,
    *,
    phase: str,
    status: str,
) -> None:
    if timing is None:
        return
    duration = max(0.0, time.monotonic() - started_at_monotonic)
    timing[metric] = float(timing.get(metric, 0.0)) + duration
    timing.setdefault("timing_events", []).append(
        {
            "phase": phase,
            "started_at_unix": round(time.time() - duration, 3),
            "duration_seconds": round(duration, 6),
            "status": status,
        }
    )

# NeMo RL sends the task transport payload directly to this agent's
# ``run_group`` call.  It does not use the Verifiers environment Dataset's
# formatted prompt, so the environment-level system_prompt cannot establish
# the browser-use contract for policy rollout.  Keep the instruction here,
# at the actual policy request boundary.  It adds no task examples, answers,
# or synthetic training data.
BROWSER_POLICY_SYSTEM_PROMPT = (
    "MANDATORY_BROWSER_TOOL_CALL: You are an autonomous browser agent and a "
    "real browser is already open on the task website. Your first assistant "
    "response MUST call the observe tool to inspect the current page. Do not "
    "write a final answer, claim that you cannot browse, or answer from prior "
    "knowledge before that tool call. After observing, use browser tools until "
    "browser evidence supports a concise final answer. You may reason in a "
    "<think> block, but finish that block before emitting a tool call: never "
    "place <tool_call> inside <think>. Emit exactly ONE browser tool call per "
    "assistant turn, wait for its result, then decide the next action. For a "
    "form submission, first observe, then act to fill/click, then observe or "
    "extract the resulting page before answering. An act instruction must name "
    "the exact observed control and operation, not merely the desired outcome. "
    "For a fill action, copy the literal task text from the user message; "
    "never write the placeholder phrase 'the user\'s query'. Do not say only "
    "'calculate' or 'search' to the act tool."
    " If observe returns selectors of the form [data-lex-id=lex-N], use "
    "the exact CDP action protocol: `fill [data-lex-id=lex-N] :: text`, "
    "`click [data-lex-id=lex-N]`, or `press [data-lex-id=lex-N] :: Enter`. "
    "In this DOM mode, after a successful click call observe to get the rendered result; "
    "do not call extract. For list questions, copy requested item labels exactly "
    "from the latest browser evidence rather than inferring a related product or "
    "renaming it (for example, preserve 'Genius Bar' rather than inventing "
    "'Genius Store'). Do not give a final answer after only the homepage unless "
    "the requested items are visibly present there."
)

_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def _recover_hermes_tool_calls(message: dict[str, Any]) -> None:
    """Add structured calls for Qwen/Hermes calls embedded inside thinking.

    Qwen can emit a valid Hermes ``<tool_call>`` block inside ``<think>``.  We
    retain its original text and token/logprob path unchanged, and only add the
    OpenAI-compatible metadata that Verifiers needs to dispatch the browser
    action.  This is deliberately not a string-cleaning or planner rewrite.
    """
    if message.get("role") != "assistant" or message.get("tool_calls"):
        return
    content = message.get("content")
    if not isinstance(content, str):
        return
    tool_calls: list[dict[str, Any]] = []
    for index, match in enumerate(_HERMES_TOOL_CALL_RE.finditer(content)):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not parsed.get("name"):
            continue
        arguments = parsed.get("arguments", {})
        tool_calls.append(
            {
                "id": f"call_hermes_{index}",
                "type": "function",
                "function": {
                    "name": str(parsed["name"]),
                    "arguments": (
                        arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False)
                    ),
                },
            }
        )
    if tool_calls:
        message["tool_calls"] = tool_calls


# patch verifiers to include prompt and generation token ids and logprobs for
# re-tokenization correction in replace_prefix_tokens (https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40)
async def _patched_parse_response_messages(response, message_type):
    messages = await _original_parse_response_messages(response, message_type)
    if message_type == "chat" and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                _recover_hermes_tool_calls(msg)
                if hasattr(response, "prompt_token_ids"):
                    msg["prompt_token_ids"] = response.prompt_token_ids
                if response.choices and hasattr(response.choices[0], "token_ids"):
                    msg["generation_token_ids"] = response.choices[0].token_ids
                if (
                    response.choices
                    and response.choices[0].logprobs
                    and hasattr(response.choices[0].logprobs, "content")
                    and response.choices[0].logprobs.content
                ):
                    msg["generation_log_probs"] = [t.logprob for t in response.choices[0].logprobs.content]
    return messages


_multiturn_env_module.parse_response_messages = _patched_parse_response_messages


class VerifiersNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    output: list[dict[str, Any]]
    reward: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: list = Field(default_factory=list)


class VerifiersAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: VerifiersNeMoGymResponse
    reward: float


class VLLMOpenAIClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        # BrowserEnv reads these attributes when proxy_model_to_stagehand=true.
        self.base_url = self._base_url
        self.api_key = "EMPTY"  # pragma: allowlist secret
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, client: "VLLMOpenAIClient") -> None:
            self.completions = client

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        timing = _ACTIVE_ROLLOUT_TIMING.get()
        request_started_at = time.monotonic()
        request_body: dict[str, Any] = {
            "model": kwargs.get("model", ""),
            "messages": kwargs.get("messages", []),
        }
        for key in (
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stop",
            "n",
            "tools",
            "tool_choice",
        ):
            if key in kwargs and kwargs[key] is not None:
                request_body[key] = kwargs[key]

        url = f"{self._base_url}/chat/completions"
        try:
            session = get_global_aiohttp_client()
            # The first policy turn has no preceding assistant message, so it
            # cannot be identified reliably from token-prefix metadata alone.
            # Mark it explicitly.  The colocated vLLM server uses this marker
            # only to choose the Qwen chat template; it is not sent to the
            # model and does not alter the policy prompt, tokens, or logprobs.
            async with session.post(
                url,
                json=request_body,
                headers={"X-Nemo-Policy-Rollout": "1"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Request to {url} failed with status {resp.status}: {error_text}")
                    resp.raise_for_status()
                response_dict = await resp.json()
        except Exception as e:
            logger.error(f"Exception calling {url}: {type(e).__name__}: {e}")
            if timing is not None:
                timing["policy_request_count"] = float(timing.get("policy_request_count", 0.0)) + 1.0
                _record_timing_event(
                    timing,
                    "policy_request_seconds",
                    request_started_at,
                    phase="policy_vllm_request",
                    status=f"error:{type(e).__name__}",
                )
            raise

        choice_dict = response_dict["choices"][0]
        # vLLM uses the non-OpenAI finish reason "abort" when an engine-side
        # request is length/cancel terminated.  The OpenAI SDK's ChatCompletion
        # schema rejects it before the agent can return a bounded trajectory.
        # Map it to the standard length terminal state; this preserves the
        # generated tokens/logprobs and makes the failure auditable downstream.
        if choice_dict.get("finish_reason") == "abort":
            logger.warning("vLLM finish_reason=abort mapped to length for bounded rollout")
            choice_dict["finish_reason"] = "length"
            choice_dict.setdefault("message", {}).setdefault("content", "")
        message_dict = choice_dict.get("message", {})

        prompt_token_ids = message_dict.pop("prompt_token_ids", [])
        generation_token_ids = message_dict.pop("generation_token_ids", [])
        generation_log_probs = message_dict.pop("generation_log_probs", [])

        if not generation_token_ids:
            logger.warning(
                f"No generation_token_ids in response! Full message keys were: {list(choice_dict.get('message', {}).keys())}"
            )

        if prompt_token_ids and isinstance(prompt_token_ids[0], str):
            prompt_token_ids = [int(tid) for tid in prompt_token_ids]

        if generation_token_ids and isinstance(generation_token_ids[0], str):
            generation_token_ids = [int(tid) for tid in generation_token_ids]

        if generation_token_ids and generation_log_probs:
            choice_dict["logprobs"] = {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                    for tid, lp in zip(generation_token_ids, generation_log_probs)
                ]
            }

        response = ChatCompletion.model_validate(response_dict)
        setattr(response, "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", generation_token_ids)
        if timing is not None:
            timing["policy_request_count"] = float(timing.get("policy_request_count", 0.0)) + 1.0
            _record_timing_event(
                timing,
                "policy_request_seconds",
                request_started_at,
                phase="policy_vllm_request",
                status="ok",
            )
        return response


class VerifiersAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model_name: str = Field(default="", description="Model name")

    vf_env_id: str = Field(default="", description="Verifiers environment ID")
    vf_env_args: dict = Field(default_factory=dict, description="Verifiers environment arguments")

    max_concurrent_generation: int = Field(
        default=-1, description="Max concurrent generation requests (-1 = unlimited)"
    )
    max_concurrent_scoring: int = Field(default=-1, description="Max concurrent scoring requests (-1 = unlimited)")

    max_tokens: int = Field(default=8192, description="Max tokens for generation")

    # nemo rl generation_config overrides these
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = Field(default=None, description="Verifiers environment ID")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    answer: str = Field(default="", description="Expected answer from dataset")
    task: str = Field(default="default", description="Task type from dataset")
    example_id: int | str = Field(default=0, description="Example ID from dataset")
    info: dict = Field(default_factory=dict, description="Extra info from dataset")


class VerifiersAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    envs_cache: dict[str, Any] = Field(default_factory=dict)  # vf.Environment
    openai_client_cache: dict[str, VLLMOpenAIClient] = Field(default_factory=dict)

    def _get_env(self, vf_env_id: str) -> vf.Environment:
        if vf_env_id not in self.envs_cache:
            self.envs_cache[vf_env_id] = vf.load_environment(vf_env_id, **self.config.vf_env_args)
        return self.envs_cache[vf_env_id]

    def _get_openai_client(self) -> VLLMOpenAIClient:
        cache_key = self.config.model_server.name
        if cache_key not in self.openai_client_cache:
            server_config_dict = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            model_server_url = f"http://{server_config_dict.host}:{server_config_dict.port}"

            if not model_server_url.endswith("/v1"):
                model_server_url = model_server_url.rstrip("/") + "/v1"

            self.openai_client_cache[cache_key] = VLLMOpenAIClient(base_url=model_server_url)

        return self.openai_client_cache[cache_key]

    @staticmethod
    def _as_dict(message: Any) -> dict:
        if isinstance(message, dict):
            return message
        return {
            key: getattr(message, key, None)
            for key in (
                "role",
                "content",
                "tool_calls",
                "tool_call_id",
                "id",
                "call_id",
                "name",
                "arguments",
                "function",
            )
        }

    @staticmethod
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return json.dumps(content, default=str)

    @staticmethod
    def _token_kwargs(tokens: dict | None) -> dict:
        if not tokens:
            return {}
        return {
            "prompt_token_ids": tokens.get("prompt_ids", []),
            "generation_token_ids": tokens.get("completion_ids", []),
            "generation_log_probs": tokens.get("completion_logprobs", []),
        }

    def _assistant_tokens(self, state: dict) -> list[dict | None]:
        tokens_per_turn: list[dict | None] = []
        for step in state.get("trajectory") or []:
            tokens = step.get("tokens") if isinstance(step, dict) else None
            for message in step.get("completion") or []:
                if self._as_dict(message).get("role") == "assistant":
                    tokens_per_turn.append(tokens)
        return tokens_per_turn

    def _function_call_item(self, raw_call: Any, tokens: dict | None) -> dict:
        call = self._as_dict(raw_call)
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        call_id = call.get("id") or call.get("call_id") or f"call_{id(raw_call)}"
        cls = NeMoGymResponseFunctionToolCallForTraining if tokens else NeMoGymResponseFunctionToolCall
        return cls(
            id=call_id,
            call_id=call_id,
            name=call.get("name") or function.get("name", ""),
            arguments=self._text(call.get("arguments") or function.get("arguments") or "{}"),
            status="completed",
            **self._token_kwargs(tokens),
        ).model_dump()

    def _convert_trajectory_to_output(self, state: dict) -> list:
        """Preserve the complete conversation and trainable tool-call tokens."""
        assistant_tokens = self._assistant_tokens(state)
        assistant_index = 0
        output: list[dict] = []
        for raw in state.get("completion") or []:
            message = self._as_dict(raw)
            role = message.get("role", "user")
            if role == "tool":
                call_id = message.get("tool_call_id") or f"call_{id(raw)}"
                output.append(
                    NeMoGymFunctionCallOutput(
                        call_id=call_id,
                        id=call_id,
                        output=self._text(message.get("content")),
                        status="completed",
                    ).model_dump()
                )
                continue
            if role != "assistant":
                output.append(
                    NeMoGymEasyInputMessage(
                        role=role, content=self._text(message.get("content"))
                    ).model_dump()
                )
                continue

            tokens = (
                assistant_tokens[assistant_index]
                if assistant_index < len(assistant_tokens)
                else None
            )
            assistant_index += 1
            body = self._text(message.get("content"))
            tool_calls = message.get("tool_calls") or []
            if body:
                body_tokens = None if tool_calls else tokens
                cls = (
                    NeMoGymResponseOutputMessageForTraining
                    if body_tokens
                    else NeMoGymResponseOutputMessage
                )
                output.append(
                    cls(
                        id=f"msg_{id(raw)}",
                        content=[NeMoGymResponseOutputText(text=body, annotations=[])],
                        **self._token_kwargs(body_tokens),
                    ).model_dump()
                )
            for index, tool_call in enumerate(tool_calls):
                call_tokens = tokens if index == len(tool_calls) - 1 else None
                output.append(self._function_call_item(tool_call, call_tokens))

        if not any(item.get("generation_token_ids") for item in output):
            logger.warning("Rollout produced no trainable token IDs; returning an empty marker")
            output.append(
                NeMoGymResponseOutputMessageForTraining(
                    id="msg_empty",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    prompt_token_ids=[0],
                    generation_token_ids=[0],
                    generation_log_probs=[0.0],
                ).model_dump()
            )
        return output

    def _trajectory_metrics(self, state: dict) -> dict[str, float]:
        """Expose browser behavior metrics alongside reward for TensorBoard."""
        counts = {
            "assistant_turns": 0.0,
            "tool_calls": 0.0,
            "tool_results": 0.0,
            "navigate_calls": 0.0,
            "observe_calls": 0.0,
            "act_calls": 0.0,
            "extract_calls": 0.0,
            "tool_errors": 0.0,
            "infrastructure_failures": 0.0,
            "policy_failures": 0.0,
            "tool_timeouts": 0.0,
            "valid_trajectory": 1.0,
            "episode_terminated": 0.0,
            "generation_tokens": 0.0,
            "setup_navigation_attempts": 1.0,
            "setup_navigation_retry_success": 0.0,
            "judge_empty_response": 0.0,
        }
        for step in state.get("trajectory") or []:
            tokens = step.get("tokens") if isinstance(step, dict) else None
            if tokens:
                counts["generation_tokens"] += float(
                    len(tokens.get("completion_ids") or [])
                )
        for raw in state.get("completion") or []:
            message = self._as_dict(raw)
            role = message.get("role")
            if role == "assistant":
                counts["assistant_turns"] += 1.0
                for raw_call in message.get("tool_calls") or []:
                    call = self._as_dict(raw_call)
                    function = (
                        call.get("function")
                        if isinstance(call.get("function"), dict)
                        else {}
                    )
                    name = call.get("name") or function.get("name", "")
                    counts["tool_calls"] += 1.0
                    key = f"{name}_calls"
                    if key in counts:
                        counts[key] += 1.0
            elif role == "tool":
                counts["tool_results"] += 1.0
                tool_text = self._text(message.get("content")).lstrip().lower()
                if tool_text.startswith("error"):
                    counts["tool_errors"] += 1.0
                if tool_text.startswith("error_infrastructure"):
                    counts["infrastructure_failures"] += 1.0
                if tool_text.startswith("error_policy"):
                    counts["policy_failures"] += 1.0
        guard = state.get("trajectory_guard")
        if guard is not None:
            counts["infrastructure_failures"] = float(
                getattr(guard, "infrastructure_failures", counts["infrastructure_failures"])
            )
            counts["policy_failures"] = float(
                getattr(guard, "policy_failures", counts["policy_failures"])
            )
            counts["tool_timeouts"] = float(getattr(guard, "timeouts", 0))
            counts["episode_terminated"] = float(bool(getattr(guard, "terminated", False)))
        counts["setup_navigation_attempts"] = float(
            state.get("setup_navigation_attempts", counts["setup_navigation_attempts"])
        )
        counts["setup_navigation_retry_success"] = float(
            state.get("setup_navigation_retry_success", 0.0)
        )
        counts["judge_empty_response"] = float(
            not str(state.get("webvoyager_judge_response") or "").strip()
        )
        if counts["infrastructure_failures"] > 0:
            # Keep the numeric reward contract required by GRPO, but clearly
            # separate invalid infrastructure episodes from policy reward in
            # logs/reports rather than silently treating them as ordinary 0s.
            counts["valid_trajectory"] = 0.0
        counts["no_tool_call"] = 1.0 if counts["tool_calls"] == 0 else 0.0
        return counts

    @staticmethod
    def _finalize_timing(state: dict, timing: dict[str, Any]) -> dict[str, float]:
        """Merge app, browser, and policy durations into audit/TensorBoard metrics."""
        finished_at_monotonic = time.monotonic()
        timing["rollout_finished_at_unix"] = round(time.time(), 3)
        timing["rollout_wall_seconds"] = max(
            0.0, finished_at_monotonic - float(timing["rollout_started_at_monotonic"])
        )

        guard = state.get("trajectory_guard")
        if guard is not None:
            guard_started_at = getattr(guard, "started_at", None)
            if guard_started_at is not None:
                timing["agent_to_browser_dispatch_seconds"] = max(
                    0.0, float(guard_started_at) - float(timing["rollout_started_at_monotonic"])
                )
            for key, value in dict(getattr(guard, "timings", {}) or {}).items():
                timing[key] = float(value)
            timing.setdefault("timing_events", []).extend(
                list(getattr(guard, "timing_events", []) or [])
            )

        timing["judge_seconds"] = float(state.get("judge_seconds", timing.get("judge_seconds", 0.0)))
        timing["browser_response_seconds"] = sum(
            float(timing.get(key, 0.0))
            for key in (
                "lexmount_session_create_seconds",
                "browser_attach_seconds",
                "browser_setup_navigation_seconds",
                "browser_tool_seconds",
            )
        )
        timing["browser_or_scheduler_wait_seconds"] = (
            float(timing.get("browser_slot_wait_seconds", 0.0))
            + float(timing["browser_response_seconds"])
        )
        # Epoch timestamps/events are audit metadata, not learning metrics.
        state["lexbrowser_timing"] = timing
        metric_keys = (
            "rollout_wall_seconds",
            "agent_to_browser_dispatch_seconds",
            "browser_slot_wait_seconds",
            "lexmount_session_create_seconds",
            "browser_attach_seconds",
            "browser_setup_navigation_seconds",
            "browser_tool_seconds",
            "browser_tool_count",
            "browser_response_seconds",
            "browser_or_scheduler_wait_seconds",
            "policy_request_seconds",
            "policy_request_count",
            "judge_seconds",
        )
        return {key: float(timing.get(key, 0.0)) for key in metric_keys}

    def _write_trajectory_audit(
        self,
        *,
        task_idx: int | str | None,
        rollout_input: Any,
        state: dict,
        reward: float,
        metrics: dict[str, Any],
    ) -> None:
        """Persist a small, secret-free trajectory record for RL diagnosis."""
        path = Path(
            os.environ.get(
                "LEXBROWSER_TRAJECTORY_AUDIT_LOG",
                "/workspace/LexBrowserEnv/logs/lexbrowser-grpo/trajectory_audit.jsonl",
            )
        )
        completion: list[dict[str, Any]] = []
        for raw in state.get("completion") or []:
            message = self._as_dict(raw)
            tool_calls = []
            for raw_call in message.get("tool_calls") or []:
                call = self._as_dict(raw_call)
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                tool_calls.append(
                    {
                        "name": call.get("name") or function.get("name", ""),
                        "arguments": self._text(call.get("arguments") or function.get("arguments")),
                    }
                )
            completion.append(
                {
                    "role": message.get("role"),
                    "content": self._text(message.get("content")),
                    "tool_calls": tool_calls,
                }
            )
        record = {
            "task_idx": task_idx,
            # ``vf.RolloutInput`` is a TypedDict in the installed Verifiers
            # release, not an object with attributes.  Use mapping access so
            # audit records reflect the exact task passed to JudgeRubric.
            "task_id": (rollout_input.get("info", {}) or {}).get("task_id"),
            "website": (rollout_input.get("info", {}) or {}).get("website"),
            "question": rollout_input.get("task", ""),
            "reward": reward,
            "metrics": metrics,
            "judge_response": state.get("webvoyager_judge_response"),
            "judge_transcript_chars": len(str(state.get("webvoyager_judge_transcript") or "")),
            # Setup failures never produce an agent completion, but they must
            # remain visible in the same audit stream as ordinary trajectories.
            # Keeping only a string avoids attempting to serialize aiohttp or
            # Stagehand exception objects through the NeMo Gym/Ray transport.
            "infrastructure_error": state.get("infrastructure_error"),
            # Timing has no prompts, credentials, CDP URLs, or DOM content.
            # Events make GPU-idle correlation reproducible after the run.
            "timing": state.get("lexbrowser_timing", {}),
            # Audit the actual policy prompt separately from the completion.
            # This lets us prove that browser-use instructions reached the
            # rollout worker without recording secrets or hidden token data.
            "prompt": [
                {
                    "role": self._as_dict(message).get("role"),
                    "content": self._text(self._as_dict(message).get("content")),
                }
                for message in (state.get("prompt", []) or [])
            ],
            "completion": completion,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("Could not write trajectory audit record: %s", exc)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        rollout_timing: dict[str, Any] = {
            "rollout_started_at_monotonic": time.monotonic(),
            "rollout_started_at_unix": round(time.time(), 3),
            "timing_events": [],
        }
        timing_token = _ACTIVE_ROLLOUT_TIMING.set(rollout_timing)
        try:
            vf_env_id = body.vf_env_id or self.config.vf_env_id
            vf_env = self._get_env(vf_env_id)
            task_idx = body.task_idx

            prompt_messages = []
            for item in body.responses_create_params.input or []:
                if hasattr(item, "role") and hasattr(item, "content"):
                    prompt_messages.append({"role": item.role, "content": item.content})
                elif isinstance(item, dict):
                    prompt_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

            # Ensure the action contract reaches the policy even when the
            # NeMo dataset/Responses transport supplies a separate generic
            # system message.  This does not alter assistant/tool history or
            # token metadata; it only establishes the real-browser protocol.
            if not any(
                BROWSER_POLICY_SYSTEM_PROMPT.split(":", 1)[0]
                in str(message.get("content", ""))
                for message in prompt_messages
                if message.get("role") == "system"
            ):
                insertion_index = next(
                    (
                        index + 1
                        for index, message in enumerate(prompt_messages)
                        if message.get("role") == "system"
                    ),
                    0,
                )
                prompt_messages.insert(
                    insertion_index,
                    {"role": "system", "content": BROWSER_POLICY_SYSTEM_PROMPT},
                )

            # NeMo Gym's transport can leave ``body.task`` empty while the
            # actual WebVoyager instruction is correctly present as the user
            # message in ``responses_create_params.input``.  JudgeRubric uses
            # RolloutInput.task to fill `{question}` in TASK_JUDGE_PROMPT, so
            # preserve that literal instruction rather than sending an empty
            # task to the judge.  This is data plumbing only: no task is
            # synthesized, rewritten, or answered here.
            task_text = self._text(body.task).strip()
            if not task_text:
                task_text = "\n".join(
                    self._text(message.get("content", "")).strip()
                    for message in prompt_messages
                    if message.get("role") == "user"
                    and self._text(message.get("content", "")).strip()
                )

            rollout_info = dict(body.info or {})
            rollout_input = vf.RolloutInput(
                prompt=prompt_messages,
                answer=body.answer,
                task=task_text,
                info=rollout_info,
                example_id=body.example_id,
            )

            client = self._get_openai_client()

            # This semaphore must be shared by all 64 HTTP requests in an
            # optimizer step. Creating one per request made a configured "8"
            # effectively behave as 64 concurrent Lexmount session creates.
            gen_sem = await _shared_semaphore(
                "generation", self.config.max_concurrent_generation
            )
            score_sem = await _shared_semaphore("scoring", self.config.max_concurrent_scoring)

            # prefer NeMo RL generation config set in responses_create_params https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/experience/rollouts.py#L1045-L1046
            sampling_args = {
                "max_tokens": self.config.max_tokens,
                "temperature": getattr(body.responses_create_params, "temperature", None) or self.config.temperature,
                "top_p": getattr(body.responses_create_params, "top_p", None) or self.config.top_p,
            }
            # ``gen_sem`` is acquired by Verifiers before the browser setup
            # path.  It serializes model-generation pressure for the whole
            # rollout, so waiting for it is normal scheduler queueing—not a
            # browser failure.  Do not put a wall-clock timeout around
            # ``run_group`` here: that would cancel queued requests before
            # they can create a Lexmount session and then starve them by
            # retrying into the same queue.  The environment itself owns
            # bounded session, navigation, tool, and episode timeouts.
            last_error: Exception | None = None
            states = None
            try:
                states = await vf_env.run_group(
                    group_inputs=[rollout_input],
                    client=client,
                    model=self.config.model_name,
                    gen_sampling_args=sampling_args,
                    gen_sem=gen_sem,
                    score_sem=score_sem,
                )
            except asyncio.CancelledError:
                # Preserve shutdown semantics; cancellation is not a reward.
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Infrastructure rollout failed for task %s without wrapper retry: %r",
                    task_idx,
                    exc,
                )
            if states is None:
                # Serialize a structured, explicitly invalid infrastructure
                # outcome instead of returning HTTP 500 with an aiohttp error
                # object that Ray cannot pickle.  Downstream reporting excludes
                # valid_trajectory=0 from reward curves.
                logger.error(
                    "Infrastructure rollout failed for task %s: %r",
                    task_idx,
                    last_error,
                )
                failure_metrics = {
                    "infrastructure_failures": 1.0,
                    "valid_trajectory": 0.0,
                    # This adapter did not itself retry setup.  Environment
                    # metrics report any real setup attempts separately.
                    "setup_navigation_attempts": 0.0,
                    "setup_navigation_retry_success": 0.0,
                }
                failure_state = {
                    "prompt": prompt_messages,
                    "completion": [],
                    "infrastructure_error": (
                        f"{type(last_error).__name__}: {last_error}"
                        if last_error is not None
                        else "unknown infrastructure failure"
                    ),
                }
                failure_metrics.update(self._finalize_timing(failure_state, rollout_timing))
                self._write_trajectory_audit(
                    task_idx=task_idx,
                    rollout_input=rollout_input,
                    state=failure_state,
                    reward=0.0,
                    metrics=failure_metrics,
                )
                output = [
                    NeMoGymResponseOutputMessageForTraining(
                        id="msg_infrastructure_failure",
                        content=[NeMoGymResponseOutputText(text="", annotations=[])],
                        prompt_token_ids=[0],
                        generation_token_ids=[0],
                        generation_log_probs=[0.0],
                    ).model_dump()
                ]
                response_payload = VerifiersNeMoGymResponse(
                    id=f"verifiers-{vf_env_id}-{task_idx}",
                    created_at=0,
                    model=self.config.model_name,
                    object="response",
                    output=output,
                    env_id=vf_env_id,
                    group_id=str(task_idx),
                    reward=0.0,
                    metrics=failure_metrics,
                )
                _ACTIVE_ROLLOUT_TIMING.reset(timing_token)
                return response_payload

            state = states[0]
            reward = state.get("reward", 0.0) or 0.0
            metrics = dict(state.get("metrics", {}) or {})
            metrics.update(self._trajectory_metrics(state))
            metrics.update(self._finalize_timing(state, rollout_timing))

            self._write_trajectory_audit(
                task_idx=task_idx,
                rollout_input=rollout_input,
                state=state,
                reward=float(reward),
                metrics=metrics,
            )

            output = self._convert_trajectory_to_output(state)

            response_payload = VerifiersNeMoGymResponse(
                id=f"verifiers-{vf_env_id}-{task_idx}",
                created_at=0,
                model=self.config.model_name,
                object="response",
                output=output,
                env_id=vf_env_id,
                group_id=str(task_idx),
                reward=reward,
                metrics=metrics,
            )
            _ACTIVE_ROLLOUT_TIMING.reset(timing_token)
            return response_payload
        except Exception as e:
            logger.error(f"Exception in responses(): {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            _ACTIVE_ROLLOUT_TIMING.reset(timing_token)
            raise

    async def run(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersAgentVerifyResponse:
        resp = await self.responses(request, response, body)

        return VerifiersAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=resp,
            reward=resp.reward,
        )


if __name__ == "__main__":
    VerifiersAgent.run_webserver()
