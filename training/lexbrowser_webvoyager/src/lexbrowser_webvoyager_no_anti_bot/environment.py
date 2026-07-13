"""WebVoyager-no-anti-bot with Lexmount Browser as the browser backend."""

# The judge prompt is intentionally byte-for-byte equivalent to the upstream package.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset
from lexmount import Lexmount
from openai import AsyncOpenAI
from stagehand import AsyncStagehand

LOGGER = logging.getLogger(__name__)

EXPECTED_DATASET_ROWS = 600
EXPECTED_DATASET_SHA256 = "b901adc3f1fb93c069260e1940c59b214374f0ffe58ff7dcf5b1af831d3b1097"
TOOL_RESULT_CHAR_LIMIT = 160
TRANSCRIPT_CHAR_LIMIT = 12_000
TRUNCATION_MARKER = "\n[...truncated...]\n"

TASK_JUDGE_PROMPT = """You are evaluating whether a browser automation agent successfully completed a web task.

Use only the browser interaction transcript as evidence. Do not assume the task was completed unless the transcript shows it.
Treat assistant claims such as "done", "completed", or quoted answers as evidence only when they are supported by the recorded tool calls and tool results.
If the transcript shows no meaningful browser interaction, the wrong website or page, missing required actions, or insufficient evidence that the requested end state was reached, respond "no".
If the transcript is truncated and the available evidence does not clearly prove success, respond "no".

Task Description:
```
{question}
```

Browser Interaction Transcript:
```
{response}
```

Based on the transcript, decide whether the task was successfully completed.

Consider:
1. Did the agent navigate to the correct website/page?
2. Did the agent perform the required actions (search, filter, click, fill forms, etc.)?
3. Did the agent reach a state that satisfies the task requirements?
4. Did the agent provide the requested information if applicable?

Respond with exactly "yes" or "no"."""


def _tool_call_field(tool_call: Any, field: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(field, default)
    return getattr(tool_call, field, default)


def _message_field(message: Any, field: str, default: Any = None) -> Any:
    """Read OpenAI messages without relying on Verifiers private APIs."""
    if isinstance(message, dict):
        return message.get(field, default)
    return getattr(message, field, default)


def _content_to_text(content: Any) -> str:
    """Normalize string and Responses-style content blocks across VF versions."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if text:
                parts.append(str(text))
        else:
            text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return json.dumps(parsed, ensure_ascii=True, sort_keys=True)
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=True, sort_keys=True)


def _truncate_text(text: str, limit: int, marker: str, *, prefix_ratio: float = 0.2) -> str:
    if len(text) <= limit:
        return text
    available = limit - len(marker)
    if available <= 0:
        return marker[:limit]
    prefix_len = max(1, int(available * prefix_ratio))
    suffix_len = max(0, available - prefix_len)
    prefix = text[:prefix_len].rstrip()
    suffix = text[-suffix_len:].lstrip() if suffix_len else ""
    return f"{prefix}{marker}{suffix}"


def _has_tool_calls(completion: str | list) -> bool:
    if not isinstance(completion, list):
        return False
    for message in completion:
        if _message_field(message, "role") != "assistant":
            continue
        tool_calls = _message_field(message, "tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def render_webvoyager_transcript(
    completion: str | list,
    *,
    tool_result_char_limit: int = TOOL_RESULT_CHAR_LIMIT,
    transcript_char_limit: int = TRANSCRIPT_CHAR_LIMIT,
) -> str:
    if isinstance(completion, str):
        return _truncate_text(completion.strip(), transcript_char_limit, TRUNCATION_MARKER)

    transcript_lines: list[str] = []
    for message in completion:
        role = _message_field(message, "role")
        if role == "assistant":
            content = _message_field(message, "content")
            assistant_text = _content_to_text(content).strip()
            if assistant_text:
                transcript_lines.append(f"ASSISTANT: {assistant_text}")
            tool_calls = _message_field(message, "tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    name = _tool_call_field(tool_call, "name", "unknown_tool")
                    args = _normalize_tool_arguments(
                        _tool_call_field(tool_call, "arguments", "{}")
                    )
                    transcript_lines.append(f"TOOL_CALL {name}({args})")
        elif role == "tool":
            content = _message_field(message, "content")
            tool_text = _content_to_text(content).strip()
            if tool_text:
                truncated = _truncate_text(
                    tool_text, tool_result_char_limit, " [...truncated]", prefix_ratio=1.0
                )
                transcript_lines.append(f"TOOL_RESULT: {truncated}")

    transcript = "\n".join(transcript_lines).strip()
    return _truncate_text(transcript, transcript_char_limit, TRUNCATION_MARKER) if transcript else ""


class WebVoyagerTrajectoryParser(vf.Parser):
    def parse_answer(self, completion: vf.Messages | str) -> str | None:
        rendered = (
            self.parse(completion)
            if isinstance(completion, str)
            else render_webvoyager_transcript(completion)
        )
        return rendered or None


async def judge_task_completion(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
    parser: vf.Parser,
) -> float:
    """Use the upstream BrowserEnv binary reward and no-tool-call short circuit."""
    if not _has_tool_calls(completion):
        return 0.0
    rendered_completion = parser.parse_answer(completion)
    if not rendered_completion:
        return 0.0
    state["webvoyager_judge_transcript"] = rendered_completion
    judge_response = await judge(prompt, rendered_completion, answer, state)
    return 1.0 if re.match(r"^\s*yes\b", judge_response, flags=re.IGNORECASE) else 0.0


@dataclass
class TrajectoryGuard:
    """Per-episode circuit breaker and auditable failure classification."""

    per_tool_timeout_s: float
    episode_timeout_s: float
    max_repeated_tool_calls: int
    started_at: float = field(default_factory=time.monotonic)
    last_signature: str = ""
    repeated_tool_calls: int = 0
    policy_failures: int = 0
    infrastructure_failures: int = 0
    timeouts: int = 0
    terminated: bool = False
    termination_reason: str = ""

    def before_tool(self, name: str, payload: str) -> str | None:
        if self.terminated:
            return self.termination_reason
        if time.monotonic() - self.started_at > self.episode_timeout_s:
            self.infrastructure_failures += 1
            self.timeouts += 1
            self.terminated = True
            self.termination_reason = "infrastructure_episode_timeout"
            return self.termination_reason
        signature = f"{name}:{payload.strip()}"
        self.repeated_tool_calls = (
            self.repeated_tool_calls + 1 if signature == self.last_signature else 0
        )
        self.last_signature = signature
        if self.repeated_tool_calls >= self.max_repeated_tool_calls:
            self.policy_failures += 1
            self.terminated = True
            self.termination_reason = "policy_no_progress_repeated_tool_call"
            return self.termination_reason
        return None

    def infrastructure_timeout(self, name: str) -> str:
        self.infrastructure_failures += 1
        self.timeouts += 1
        self.terminated = True
        self.termination_reason = f"infrastructure_{name}_timeout"
        return self.termination_reason


def load_webvoyager_dataset(num_examples: int = -1, web_filter: str | None = None) -> Dataset:
    dataset_path = Path(__file__).parent / "datasets" / "WebVoyager_data_clean.jsonl"
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if digest != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Unexpected WebVoyager dataset SHA256: {digest}; "
            f"expected {EXPECTED_DATASET_SHA256}"
        )
    examples: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if web_filter and item.get("web_name") != web_filter:
                continue
            examples.append(
                {
                    "question": item["ques"],
                    "answer": "",
                    "start_url": item["web"],
                    "task_id": item["id"],
                    "website": item["web_name"],
                    "info": {
                        "start_url": item["web"],
                        "task_id": item["id"],
                        "website": item["web_name"],
                    },
                }
            )
    if not web_filter and len(examples) != EXPECTED_DATASET_ROWS:
        raise ValueError(
            f"Unexpected WebVoyager row count: {len(examples)}; "
            f"expected {EXPECTED_DATASET_ROWS}"
        )
    if num_examples > 0:
        examples = examples[:num_examples]
    return Dataset.from_list(examples)


class LexmountDOMMode:
    """Stagehand DOM mode attached to a Lexmount-provisioned Chrome over CDP."""

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str,
        stagehand_model: str,
        policy_model: str,
        proxy_model_to_stagehand: bool,
        browser_mode: str,
        official_proxy: bool,
        max_concurrent_sessions: int,
        stagehand_ready_timeout_s: float,
        per_tool_timeout_s: float,
        episode_timeout_s: float,
        max_repeated_tool_calls: int,
    ) -> None:
        self.lexmount = Lexmount(api_key=api_key, project_id=project_id)
        self.stagehand_model = stagehand_model
        self.policy_model = policy_model
        self.proxy_model_to_stagehand = proxy_model_to_stagehand
        self.browser_mode = browser_mode
        self.official_proxy = official_proxy
        self.stagehand_ready_timeout_s = stagehand_ready_timeout_s
        self.per_tool_timeout_s = per_tool_timeout_s
        self.episode_timeout_s = episode_timeout_s
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.stagehand_client: AsyncStagehand | None = None
        self._client_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_concurrent_sessions)
        self.logger = LOGGER

    def register_tools(self, env: vf.StatefulToolEnv) -> None:
        self.logger = env.logger
        env.add_tool(self.navigate, args_to_skip=["session", "guard"])
        env.add_tool(self.observe, args_to_skip=["session", "llm_config", "guard"])
        env.add_tool(self.act, args_to_skip=["session", "llm_config", "guard"])
        env.add_tool(self.extract, args_to_skip=["session", "llm_config", "guard"])

    async def _get_stagehand_client(self) -> AsyncStagehand:
        async with self._client_lock:
            if self.stagehand_client is None:
                self.stagehand_client = AsyncStagehand(
                    server="local",
                    browserbase_api_key="local-cdp-not-browserbase",
                    model_api_key=os.environ.get("MODEL_API_KEY") or os.environ["OPENAI_API_KEY"],
                    local_ready_timeout_s=self.stagehand_ready_timeout_s,
                )
        return self.stagehand_client

    async def setup_state(self, state: vf.State) -> vf.State:
        await self._slots.acquire()
        state["lexbrowser_slot_acquired"] = True
        state["trajectory_guard"] = TrajectoryGuard(
            per_tool_timeout_s=self.per_tool_timeout_s,
            episode_timeout_s=self.episode_timeout_s,
            max_repeated_tool_calls=self.max_repeated_tool_calls,
        )
        lexmount_session = None
        try:
            lexmount_session = await asyncio.to_thread(
                self.lexmount.sessions.create,
                browser_mode=self.browser_mode,
                official_proxy=self.official_proxy,
            )
            cdp_url = lexmount_session.connect_url
            if not cdp_url:
                raise RuntimeError("Lexmount session did not return a CDP URL")
            client = await self._get_stagehand_client()
            stagehand_session = await client.sessions.start(
                model_name=self.stagehand_model,
                browser={"type": "local", "cdp_url": cdp_url},
                verbose=0,
            )
            state["lexmount_session"] = lexmount_session
            state["lexmount_session_id"] = lexmount_session.id
            state["stagehand_session"] = stagehand_session
            state["stagehand_session_id"] = stagehand_session.id
            return state
        except Exception:
            if lexmount_session is not None:
                await asyncio.to_thread(lexmount_session.close)
            self._release_slot(state)
            raise

    def _llm_config(self, state: vf.State) -> dict[str, str] | None:
        if not self.proxy_model_to_stagehand:
            return None
        wrapper = state.get("client")
        native_client = getattr(wrapper, "client", wrapper)
        base_url = str(
            getattr(native_client, "base_url", "")
            or getattr(native_client, "_base_url", "")
            or ""
        ).rstrip("/")
        api_key = str(getattr(native_client, "api_key", "") or "")
        if not base_url:
            return None
        return {
            "modelName": self.policy_model,
            "baseURL": base_url,
            "apiKey": api_key or "EMPTY",
        }

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del messages, kwargs
        updated = dict(tool_args)
        if tool_name in {"navigate", "observe", "act", "extract"}:
            updated["session"] = state["stagehand_session"]
            updated["guard"] = state["trajectory_guard"]
        if tool_name in {"observe", "act", "extract"}:
            updated["llm_config"] = self._llm_config(state)
        return updated

    def _release_slot(self, state: vf.State) -> None:
        if state.pop("lexbrowser_slot_acquired", False):
            self._slots.release()

    async def cleanup_session(self, state: vf.State) -> None:
        try:
            stagehand_session = state.pop("stagehand_session", None)
            if stagehand_session is not None:
                try:
                    await stagehand_session.end()
                except Exception as exc:
                    self.logger.warning("Failed to end Stagehand session: %s", exc)
            lexmount_session = state.pop("lexmount_session", None)
            if lexmount_session is not None:
                await asyncio.to_thread(lexmount_session.close)
        finally:
            state.pop("stagehand_session_id", None)
            state.pop("lexmount_session_id", None)
            self._release_slot(state)

    async def teardown(self) -> None:
        if self.stagehand_client is not None:
            await self.stagehand_client.close()
            self.stagehand_client = None

    async def navigate(self, url: str, session: Any, guard: TrajectoryGuard) -> str:
        """Navigate the browser directly to a URL."""
        blocked = guard.before_tool("navigate", url)
        if blocked:
            return f"ERROR_{blocked.upper()}: trajectory terminated"
        try:
            await asyncio.wait_for(
                session.navigate(url=url), timeout=guard.per_tool_timeout_s
            )
            return f"Navigated to {url}"
        except asyncio.TimeoutError:
            return f"ERROR_{guard.infrastructure_timeout('navigate').upper()}: retryable browser timeout"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_NAVIGATE: {exc}"

    async def observe(
        self, instruction: str, session: Any, llm_config: Any = None, guard: TrajectoryGuard | None = None
    ) -> str:
        """Find possible page actions matching an instruction."""
        assert guard is not None
        blocked = guard.before_tool("observe", instruction)
        if blocked:
            return f"ERROR_{blocked.upper()}: trajectory terminated"
        try:
            if llm_config:
                response = await asyncio.wait_for(session.observe(
                    instruction=instruction, options={"model": llm_config}
                ), timeout=guard.per_tool_timeout_s)
            else:
                response = await asyncio.wait_for(
                    session.observe(instruction=instruction), timeout=guard.per_tool_timeout_s
                )
            actions = [
                {
                    "description": action.description,
                    "selector": action.selector,
                    "method": action.method,
                }
                for action in response.data.result
            ]
            return json.dumps(actions, indent=2) if actions else "No matching elements found"
        except asyncio.TimeoutError:
            return f"ERROR_{guard.infrastructure_timeout('observe').upper()}: retryable browser timeout"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_OBSERVE: {exc}"

    async def act(
        self, instruction: str, session: Any, llm_config: Any = None, guard: TrajectoryGuard | None = None
    ) -> str:
        """Perform one natural-language action on the current page."""
        assert guard is not None
        blocked = guard.before_tool("act", instruction)
        if blocked:
            return f"ERROR_{blocked.upper()}: trajectory terminated"
        try:
            if llm_config:
                response = await asyncio.wait_for(session.act(
                    input=instruction, options={"model": llm_config}
                ), timeout=guard.per_tool_timeout_s)
            else:
                response = await asyncio.wait_for(
                    session.act(input=instruction), timeout=guard.per_tool_timeout_s
                )
            result = response.data.result
            return f"{'Success' if result.success else 'Failed'}: {result.message}"
        except asyncio.TimeoutError:
            return f"ERROR_{guard.infrastructure_timeout('act').upper()}: retryable browser timeout"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_ACT: {exc}"

    async def extract(
        self,
        instruction: str,
        schema_json: str,
        session: Any,
        llm_config: Any = None,
        guard: TrajectoryGuard | None = None,
    ) -> str:
        """Extract structured data from the current page using a JSON schema."""
        assert guard is not None
        blocked = guard.before_tool("extract", instruction)
        if blocked:
            return f"ERROR_{blocked.upper()}: trajectory terminated"
        try:
            schema = json.loads(schema_json)
            if llm_config:
                response = await asyncio.wait_for(session.extract(
                    instruction=instruction,
                    schema=schema,
                    options={"model": llm_config},
                ), timeout=guard.per_tool_timeout_s)
            else:
                response = await asyncio.wait_for(session.extract(
                    instruction=instruction, schema=schema
                ), timeout=guard.per_tool_timeout_s)
            return json.dumps(response.data.result, indent=2)
        except json.JSONDecodeError as exc:
            guard.policy_failures += 1
            return f"Error parsing schema JSON: {exc}"
        except asyncio.TimeoutError:
            return f"ERROR_{guard.infrastructure_timeout('extract').upper()}: retryable browser timeout"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_EXTRACT: {exc}"


class LexBrowserEnv(vf.StatefulToolEnv):
    def __init__(self, *, mode_impl: LexmountDOMMode, **kwargs: Any) -> None:
        super().__init__(stop_errors=[vf.SandboxError], **kwargs)
        self._mode_impl = mode_impl
        self._mode_impl.register_tools(self)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await self._mode_impl.setup_state(state)
        info = state.get("info") or {}
        start_url = info.get("start_url") or state.get("start_url")
        if start_url:
            # BrowserEnv reset semantics: WebVoyager begins on the dataset's
            # website. This navigation is environment setup, not an agent tool call.
            await state["stagehand_session"].navigate(url=start_url)
            state["initial_url"] = start_url
        initialized = await super().setup_state(state, **kwargs)
        return initialized if initialized is not None else state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._mode_impl.update_tool_args(tool_name, tool_args, messages, state, **kwargs)

    @vf.cleanup
    async def cleanup_session(self, state: vf.State) -> None:
        await self._mode_impl.cleanup_session(state)

    @vf.teardown
    async def teardown(self) -> None:
        await self._mode_impl.teardown()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_environment(
    mode: str = "dom",
    max_turns: int = 30,
    judge_model: str = "glm-5.2",
    num_examples: int = -1,
    web_filter: str | None = None,
    stagehand_model: str = "openai/glm-5.2",
    policy_model: str = "gpt-4o",
    proxy_model_to_stagehand: bool = True,
    browser_mode: str = "normal",
    official_proxy: bool = False,
    max_concurrent_sessions: int = 20,
    stagehand_ready_timeout_s: float = 60.0,
    per_tool_timeout_s: float = 25.0,
    episode_timeout_s: float = 180.0,
    max_repeated_tool_calls: int = 3,
    **kwargs: Any,
) -> vf.Environment:
    if mode != "dom":
        raise ValueError("lexbrowser/webvoyager-no-anti-bot currently supports mode='dom' only")

    dataset = load_webvoyager_dataset(num_examples=num_examples, web_filter=web_filter)
    parser = WebVoyagerTrajectoryParser()
    judge_client = AsyncOpenAI(
        api_key=os.environ.get("JUDGE_API_KEY") or _required_env("OPENAI_API_KEY"),
        base_url=os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )
    rubric = vf.JudgeRubric(
        parser=parser,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=TASK_JUDGE_PROMPT,
        judge_sampling_args={"temperature": 0.0, "max_tokens": 8},
    )
    rubric.add_reward_func(judge_task_completion, weight=1.0)

    mode_impl = LexmountDOMMode(
        api_key=_required_env("LEXMOUNT_API_KEY"),
        project_id=_required_env("LEXMOUNT_PROJECT_ID"),
        stagehand_model=stagehand_model,
        policy_model=policy_model,
        proxy_model_to_stagehand=proxy_model_to_stagehand,
        browser_mode=browser_mode,
        official_proxy=official_proxy,
        max_concurrent_sessions=max_concurrent_sessions,
        stagehand_ready_timeout_s=stagehand_ready_timeout_s,
        per_tool_timeout_s=per_tool_timeout_s,
        episode_timeout_s=episode_timeout_s,
        max_repeated_tool_calls=max_repeated_tool_calls,
    )
    return LexBrowserEnv(
        mode_impl=mode_impl,
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        max_turns=max_turns,
        **kwargs,
    )
