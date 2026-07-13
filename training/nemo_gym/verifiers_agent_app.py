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

import logging
import traceback
import json
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


# patch verifiers to include prompt and generation token ids and logprobs for
# re-tokenization correction in replace_prefix_tokens (https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40)
async def _patched_parse_response_messages(response, message_type):
    messages = await _original_parse_response_messages(response, message_type)
    if message_type == "chat" and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
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
            async with session.post(url, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Request to {url} failed with status {resp.status}: {error_text}")
                    resp.raise_for_status()
                response_dict = await resp.json()
        except Exception as e:
            logger.error(f"Exception calling {url}: {type(e).__name__}: {e}")
            raise

        choice_dict = response_dict["choices"][0]
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
            "generation_tokens": 0.0,
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
                if self._text(message.get("content")).lstrip().lower().startswith("error"):
                    counts["tool_errors"] += 1.0
        counts["no_tool_call"] = 1.0 if counts["tool_calls"] == 0 else 0.0
        return counts

    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
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

            rollout_input = vf.RolloutInput(
                prompt=prompt_messages,
                answer=body.answer,
                task=body.task,
                info=body.info,
                example_id=body.example_id,
            )

            client = self._get_openai_client()

            gen_sem = await maybe_semaphore(self.config.max_concurrent_generation)
            score_sem = await maybe_semaphore(self.config.max_concurrent_scoring)

            # prefer NeMo RL generation config set in responses_create_params https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/experience/rollouts.py#L1045-L1046
            sampling_args = {
                "max_tokens": self.config.max_tokens,
                "temperature": getattr(body.responses_create_params, "temperature", None) or self.config.temperature,
                "top_p": getattr(body.responses_create_params, "top_p", None) or self.config.top_p,
            }
            states = await vf_env.run_group(
                group_inputs=[rollout_input],
                client=client,
                model=self.config.model_name,
                gen_sampling_args=sampling_args,
                gen_sem=gen_sem,
                score_sem=score_sem,
            )

            state = states[0]
            reward = state.get("reward", 0.0) or 0.0
            metrics = dict(state.get("metrics", {}) or {})
            metrics.update(self._trajectory_metrics(state))

            output = self._convert_trajectory_to_output(state)

            return VerifiersNeMoGymResponse(
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
        except Exception as e:
            logger.error(f"Exception in responses(): {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
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
