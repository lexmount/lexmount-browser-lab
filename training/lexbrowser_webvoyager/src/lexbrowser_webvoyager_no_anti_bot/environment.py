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
import websocket

LOGGER = logging.getLogger(__name__)

EXPECTED_DATASET_ROWS = 600
EXPECTED_DATASET_SHA256 = "b901adc3f1fb93c069260e1940c59b214374f0ffe58ff7dcf5b1af831d3b1097"
# The judge must see the rendered evidence that the agent based its answer on.
# 160 characters only contains the URL/header of a real-page observation and
# hides the relevant controls (for example Apple "Genius Bar" / "Repair"),
# systematically turning supported answers into reward=0.  This applies only
# to the out-of-band judge transcript, never the policy context or token path.
TOOL_RESULT_CHAR_LIMIT = 1024
TRANSCRIPT_CHAR_LIMIT = 12_000
TRUNCATION_MARKER = "\n[...truncated...]\n"
SETUP_NAVIGATION_MAX_ATTEMPTS = 3

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

# This is the agent-side contract, not task content.  Verifiers formats the
# environment's Dataset itself, so a system message placed only in NeMo Gym's
# transport JSONL is not guaranteed to reach the policy.  Supplying it through
# ``Environment(system_prompt=...)`` makes it part of every real WebVoyager
# rollout without changing any of the 600 upstream tasks.
BROWSER_AGENT_SYSTEM_PROMPT = """You are an autonomous browser agent. A real browser is already open on the task website.
You must complete the user's task with the provided browser tools; do not answer from prior knowledge and never say that you cannot browse.
First call observe to inspect the current page, then use observe, act, navigate, and extract as needed. Keep using tools until browser evidence supports a concise final answer."""

CDP_BROWSER_AGENT_SYSTEM_PROMPT = """You are an autonomous browser agent. A real browser is already open on the task website.
You must complete the user's task with the provided browser tools; do not answer from prior knowledge and never say that you cannot browse.
First call observe. It returns explicit `[data-lex-id=lex-N]` selectors. For every act call, use exactly one of: `fill [data-lex-id=lex-N] :: text`, `click [data-lex-id=lex-N]`, or `press [data-lex-id=lex-N] :: Enter`. Observe again after actions and only answer with browser evidence."""


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
    started_at = time.monotonic()
    if not _has_tool_calls(completion):
        state["judge_seconds"] = time.monotonic() - started_at
        return 0.0
    rendered_completion = parser.parse_answer(completion)
    if not rendered_completion:
        state["judge_seconds"] = time.monotonic() - started_at
        return 0.0
    try:
        state["webvoyager_judge_transcript"] = rendered_completion
        judge_response = await judge(prompt, rendered_completion, answer, state)
        # Preserve the binary judge's literal response for the local, secret-free
        # rollout audit.  It makes reward=0 diagnosable without changing the
        # reward contract or exposing the judge prompt/API credentials.
        state["webvoyager_judge_response"] = str(judge_response)
        return 1.0 if re.match(r"^\s*yes\b", judge_response, flags=re.IGNORECASE) else 0.0
    finally:
        state["judge_seconds"] = time.monotonic() - started_at


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
    timings: dict[str, float] = field(default_factory=dict)
    timing_events: list[dict[str, Any]] = field(default_factory=list)

    def record_timing(
        self,
        metric: str,
        started_at: float,
        *,
        phase: str | None = None,
        status: str = "ok",
    ) -> float:
        """Record a secret-free wall-clock interval for rollout diagnosis.

        This deliberately lives in the environment guard rather than NeMo-RL:
        it measures provider/browser work only and cannot alter policy tokens,
        rewards, or scheduling behavior.
        """
        duration = max(0.0, time.monotonic() - started_at)
        self.timings[metric] = self.timings.get(metric, 0.0) + duration
        if phase is not None:
            self.timing_events.append(
                {
                    "phase": phase,
                    "started_at_unix": round(time.time() - duration, 3),
                    "duration_seconds": round(duration, 6),
                    "status": status,
                }
            )
        return duration

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


class LexmountCDPSession:
    """Small synchronous CDP adapter used only by the optional deterministic DOM mode.

    It is deliberately browser-only: it does not call an LLM or infer task
    answers.  This lets a Web RL policy consume an inspected DOM and issue
    explicit selector actions when the on-policy model cannot serve Stagehand's
    JSON-only helper protocol.
    """

    def __init__(self, cdp_url: str, timeout_s: float) -> None:
        self._ws = websocket.create_connection(
            cdp_url,
            timeout=timeout_s,
            http_proxy_host=None,
            http_proxy_port=None,
        )
        self._next_id = 0
        targets = self.call("Target.getTargets").get("targetInfos", [])
        page = next((item for item in targets if item.get("type") == "page"), None)
        if page is None:
            target_id = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        else:
            target_id = page["targetId"]
        self._session_id = self.call(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )["sessionId"]
        self.call("Page.enable", session_id=self._session_id)
        self.call("Runtime.enable", session_id=self._session_id)
        self._observed_controls: list[dict[str, str]] = []
        self._task_query = ""

    def set_task_query(self, query: str) -> None:
        """Expose the current user request only for an explicit policy placeholder.

        This does not choose an action or transform a task into a browser plan:
        it is the same literal user text that the policy received.  It merely
        makes BrowserEnv's common ``the user's query`` action wording usable
        without an LLM helper in direct-CDP mode.
        """
        self._task_query = query

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def call(
        self, method: str, params: dict[str, Any] | None = None, *, session_id: str | None = None
    ) -> dict[str, Any]:
        self._next_id += 1
        request: dict[str, Any] = {"id": self._next_id, "method": method, "params": params or {}}
        if session_id:
            request["sessionId"] = session_id
        self._ws.send(json.dumps(request))
        while True:
            response = json.loads(self._ws.recv())
            if response.get("id") != self._next_id:
                continue
            if "error" in response:
                raise RuntimeError(f"CDP {method} failed: {response['error']}")
            return response.get("result", {})

    def navigate(self, url: str) -> None:
        self.call("Page.navigate", {"url": url}, session_id=self._session_id)

    def wait_for_usable_document(self, timeout_s: float) -> None:
        """Confirm that a setup navigation yielded a real, usable document.

        ``Page.navigate`` returning only means Chrome accepted the command. In
        a remote-browser deployment it can subsequently land on
        ``chrome-error://`` or remain an empty interstitial. Detect that while
        still in environment setup so a fresh Lexmount session is tried rather
        than exposing an infrastructure error as the policy's first observe.
        """
        deadline = time.monotonic() + timeout_s
        last_url = ""
        last_text = ""
        while time.monotonic() < deadline:
            raw = self.evaluate(
                """(() => JSON.stringify({
                  url: location.href,
                  text: (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 300)
                }))()"""
            )
            snapshot = json.loads(raw or "{}")
            last_url = str(snapshot.get("url") or "")
            last_text = str(snapshot.get("text") or "")
            if (
                last_url.startswith("chrome-error://")
                or re.search(r"\bERR_[A-Z_]+\b", last_text)
            ):
                reason = re.search(r"\bERR_[A-Z_]+\b", last_text)
                raise RuntimeError(
                    "infrastructure_browser_error_page: "
                    f"{reason.group(0) if reason else last_url}"
                )
            if last_url and last_url != "about:blank" and len(last_text) >= 20:
                if re.search(
                    r"cloudflare|verify you are human|checking your browser|just a moment",
                    last_text,
                    flags=re.IGNORECASE,
                ):
                    raise RuntimeError("infrastructure_anti_bot_challenge")
                return
            time.sleep(0.25)
        raise TimeoutError(
            "infrastructure_setup_document_not_ready "
            f"url={last_url or 'about:blank'} text_chars={len(last_text)}"
        )

    def evaluate(self, expression: str) -> Any:
        response = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            session_id=self._session_id,
        )
        result = response.get("result", {})
        if result.get("subtype") == "error":
            raise RuntimeError(result.get("description", "CDP evaluation failed"))
        return result.get("value")

    def observe(self) -> dict[str, Any]:
        # Assign short, page-local ids on every observation.  The policy sees
        # these selectors and must feed one back to ``act``; no task planner is
        # embedded in this browser adapter.
        raw = self.evaluate(
            """(() => JSON.stringify({
              url: location.href,
              visible_text: (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 1600),
              elements: Array.from(document.querySelectorAll(
              'input,textarea,select,button,a,[role=button],[contenteditable=true]'
            )).filter(el => { const r = el.getBoundingClientRect(); return r.width > 1 && r.height > 1; })
            // Keep enough controls for expanded site menus (Apple's Repair
            // link comes after the primary product navigation), while the
            // compact response still leaves room for multi-turn 12K rollout.
            .slice(0, 32).map((el, i) => {
              const id = `lex-${i}`; el.setAttribute('data-lex-id', id);
              const label = (el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
                el.innerText || el.value || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 80);
              // An unquoted CSS identifier is valid here and avoids nested
              // quote escaping inside the policy's JSON tool arguments.
              return {selector: `[data-lex-id=${id}]`, tag: el.tagName.toLowerCase(),
                description: label || el.tagName.toLowerCase(),
                actions: /^(input|textarea|select)$/i.test(el.tagName) || el.isContentEditable ? 'fill' : 'click'};
            })
            }))()"""
        )
        snapshot = json.loads(raw or "{}")
        url = str(snapshot.get("url") or "")
        visible_text = str(snapshot.get("visible_text") or "")
        # Chrome's built-in error document is a transport/browser failure, not
        # an empty page chosen by the policy.  Surface it as infrastructure so
        # an unavailable provider proxy (for example ERR_TUNNEL_CONNECTION_FAILED)
        # cannot silently become a valid reward=0 trajectory.
        if (
            url.startswith("chrome-error://")
            or re.search(r"\bERR_[A-Z_]+\b", visible_text)
            or "this site can’t be reached" in visible_text.lower()
            or "this site can't be reached" in visible_text.lower()
        ):
            reason = re.search(r"\bERR_[A-Z_]+\b", visible_text)
            raise RuntimeError(
                "infrastructure_browser_error_page: "
                f"{reason.group(0) if reason else url}"
            )
        self._observed_controls = list(snapshot.get("elements") or [])
        return snapshot

    def visible_text(self) -> str:
        """Read rendered page evidence without a model-based extractor."""
        value = self.evaluate(
            "(document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 3000)"
        )
        return str(value or "")

    def _control_for_label(self, label: str, verb: str) -> str:
        """Resolve only a previously-observed control, without an LLM planner."""
        label = re.sub(
            r"^\s*(?:the\s+)?(?:textbox|text box|button|input|field)\s+described\s+as\s+",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip(" .")
        query_tokens = {
            token for token in re.findall(r"[a-z0-9]+", label.lower())
            if token not in {"the", "a", "an", "to", "with", "and", "field", "button", "input"}
        }
        candidates: list[tuple[float, str]] = []
        eligible: list[str] = []
        for control in self._observed_controls:
            description = str(control.get("description", "")).lower()
            selector = str(control.get("selector", ""))
            actions = str(control.get("actions", ""))
            if not selector or (verb in {"fill", "type"} and "fill" not in actions):
                continue
            eligible.append(selector)
            control_tokens = set(re.findall(r"[a-z0-9]+", description))
            normalized_description = re.sub(r"\s+", " ", description).strip()
            normalized_label = re.sub(r"\s+", " ", label.lower()).strip()
            # Websites commonly expose a camel-cased aria-label (for example
            # ``WolframAlpha input field``) while the policy writes the human
            # spelling ``Wolfram Alpha input field``.  Treat only an exact
            # alphanumeric-normalized match as equivalent; it remains a match
            # against the last observed control rather than semantic planning.
            compact_description = re.sub(r"[^a-z0-9]", "", normalized_description)
            compact_label = re.sub(r"[^a-z0-9]", "", normalized_label)
            if (
                normalized_description == normalized_label
                or compact_description == compact_label
            ):
                score = 3.0
            elif description and description in label.lower():
                score = 2.0
            elif query_tokens:
                score = len(query_tokens & control_tokens) / len(query_tokens)
            else:
                score = 0.0
            if score >= 0.6:
                candidates.append((score, selector))
        if not candidates and verb in {"fill", "type"} and not query_tokens and len(eligible) == 1:
            # A generic "input field" is grounded when the current DOM has
            # exactly one fillable control.  It is not a semantic guess.
            return eligible[0]
        if not candidates:
            raise ValueError("No previously observed control matches the requested action")
        candidates.sort(reverse=True)
        # A few production pages render the same accessible label twice (e.g.
        # desktop and mobile variants).  DOM order is deterministic and is a
        # better compatibility rule than rejecting an otherwise grounded exact
        # label.  The chosen element is still from the last observation only.
        return candidates[0][1]

    def _run_action(self, verb: str, selector: str, value: str = "") -> str:
        payload = json.dumps(value)
        selector_json = json.dumps(selector)
        before_url = ""
        before_text = ""
        if verb in {"fill", "type"}:
            expression = f"""(() => {{ const e=document.querySelector({selector_json}); if(!e) throw new Error('selector not found'); e.focus(); const p=Object.getPrototypeOf(e); const d=Object.getOwnPropertyDescriptor(p,'value'); if(d&&d.set) d.set.call(e,{payload}); else e.value={payload}; e.dispatchEvent(new InputEvent('input',{{bubbles:true,inputType:'insertText',data:{payload}}})); e.dispatchEvent(new Event('change',{{bubbles:true}})); return 'filled'; }})()"""
        elif verb == "click":
            # ``HTMLElement.click`` already runs the browser's native default
            # action (including form submission for a real submit control).
            # Calling ``requestSubmit(e)`` afterwards is both redundant and
            # invalid for non-submit controls; on some real sites it resets a
            # successfully navigated form.  Dispatch pointer events for React
            # handlers, then let the native click be the sole submission path.
            before_url = str(self.evaluate("location.href") or "")
            before_text = self.visible_text()
            expression = f"""(() => {{ const e=document.querySelector({selector_json}); if(!e) throw new Error('selector not found'); e.focus(); for(const t of ['pointerdown','mousedown','pointerup','mouseup']) e.dispatchEvent(new MouseEvent(t,{{bubbles:true,cancelable:true,view:window}})); e.click(); return 'clicked'; }})()"""
        else:
            key = json.dumps(value or "Enter")
            expression = f"""(() => {{ const e=document.querySelector({selector_json}); if(!e) throw new Error('selector not found'); e.focus(); e.dispatchEvent(new KeyboardEvent('keydown',{{key:{key},bubbles:true}})); e.dispatchEvent(new KeyboardEvent('keyup',{{key:{key},bubbles:true}})); return 'pressed'; }})()"""
        result = str(self.evaluate(expression) or "ok")
        if result == "clicked":
            # A real SPA can change location immediately while rendering its
            # result a few seconds later.  Poll the rendered DOM only; this is
            # synchronization after the policy's explicit click, not an
            # implicit browser action or semantic extraction.
            changed_at: float | None = None
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                time.sleep(0.4)
                current_url = str(self.evaluate("location.href") or "")
                current_text = self.visible_text()
                if current_url != before_url or current_text != before_text:
                    if changed_at is None:
                        changed_at = time.monotonic()
                    # Let a navigation/hydration settle, while avoiding a
                    # fixed multi-second delay for ordinary non-nav clicks.
                    if time.monotonic() - changed_at >= 1.6:
                        break
        return result

    def act(self, instruction: str) -> str:
        # Exact protocol deliberately keeps action grounding in the policy:
        # ``fill [data-lex-id=lex-0] :: query`` or ``click [data-lex-id=lex-1]``.
        match = re.match(
            r'^\s*(fill|type|click|press)\s+(\[data-lex-id=lex-\d+\])(?:\s*::\s*(.*))?\s*$',
            instruction,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return self._run_action(match.group(1).lower(), match.group(2), match.group(3) or "")

        # BrowserEnv's Stagehand mode accepts a compact natural-language act
        # instruction.  Keep this compatibility through deterministic matching
        # against the exact controls just returned by ``observe``.  It neither
        # plans the task nor calls an LLM; ambiguity is rejected instead of
        # guessing.  It additionally permits the common "fill ... and click
        # ..." compound instruction as two explicit DOM events.
        # Capture the value before falling back to generic fill parsing.  The
        # 1.7B policy commonly says ``with the expression '...'``; the former
        # generic regex treated that phrase as part of the control label and
        # rejected an otherwise literal, grounded action.
        fill = re.search(
            r'''\b(?:fill|type)\s+(?:the\s+)?(?P<label>.+?)\s+(?:with|as)\s+(?:the\s+)?(?:expression\s+)?(?P<value>["'][^"']+["']|the\s+user(?:'s)?\s+query)(?:\s+and\s+click\b|\s*[.!]?$)''',
            instruction,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not fill:
            fill = re.search(
                r'''\b(?:fill|type)\s+(?:the\s+)?(?P<label>.+?)(?:\s+(?:with|as)\s+(?P<value>["'][^"']+["']|the\s+user(?:'s)?\s+query))?(?:\s+and\s+click\b|\s*[.!]?$)''',
                instruction,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if not fill:
            # Qwen frequently emits the equally grounded BrowserGym-style
            # wording: "Enter 'text' into the <observed control>".
            enter = re.search(
                r'''\b(?:enter|type)\s+(?:the\s+)?(?:expression\s+)?(?P<value>["'][^"']+["'])\s+(?:into|in)\s+(?:the\s+)?(?P<label>.+?)(?:\s*[.!]?$)''',
                instruction,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if enter:
                fill = enter
        click = re.search(
            r'''\bclick\s+(?:the\s+)?(?P<label>.+?)(?:\s*[.!]?$)''',
            instruction,
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: list[str] = []
        if fill:
            raw_value = (fill.group("value") or "").strip()
            if raw_value.lower() in {"the user's query", "the user query"}:
                value = self._task_query
            elif len(raw_value) >= 2 and raw_value[0] in {"'", '"'} and raw_value[-1] == raw_value[0]:
                value = raw_value[1:-1]
            else:
                value = raw_value
            if not value:
                raise ValueError("Fill action requires literal text or 'the user\'s query'")
            results.append(self._run_action(
                "fill", self._control_for_label(fill.group("label"), "fill"), value
            ))
        if click:
            results.append(self._run_action(
                "click", self._control_for_label(click.group("label"), "click")
            ))
        if results:
            return ", ".join(results)

        # Stagehand accepts concise high-level verbs such as "Compute ...".
        # In direct-CDP mode this is only accepted when exactly one control in
        # the last DOM observation carries that same verb in its accessible
        # label (e.g. "Compute input button").  It is a deterministic label
        # match, not task planning.
        action_word = re.search(r"\b(compute|submit|search|send|go)\b", instruction, re.I)
        if action_word:
            word = action_word.group(1).lower()
            matching = [
                str(control.get("selector"))
                for control in self._observed_controls
                if word in str(control.get("description", "")).lower()
                and str(control.get("selector", ""))
            ]
            if len(matching) == 1:
                return self._run_action("click", matching[0])
            if word == "submit":
                # ``submit`` and a labelled Compute/Search button are the
                # same browser-form operation.  This is allowed only when
                # exactly one previously observed click control has one of
                # those literal submit labels; it never derives an action
                # from task content or chooses between candidates.
                submit_controls = [
                    str(control.get("selector"))
                    for control in self._observed_controls
                    if "click" in str(control.get("actions", ""))
                    and re.search(
                        r"\b(?:submit|compute|search|send|go)\b",
                        str(control.get("description", "")),
                        re.IGNORECASE,
                    )
                    and str(control.get("selector", ""))
                ]
                if len(submit_controls) == 1:
                    return self._run_action("click", submit_controls[0])
        raise ValueError(
            "Invalid CDP action. Use `fill [data-lex-id=lex-N] :: text`, "
            "`click [data-lex-id=lex-N]`, or name one previously-observed control."
        )


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
        dom_backend: str,
        stagehand_model: str,
        policy_model: str,
        proxy_model_to_stagehand: bool,
        browser_mode: str,
        official_proxy: bool,
        external_proxy: dict[str, str] | None,
        max_concurrent_sessions: int,
        session_create_timeout_s: float,
        stagehand_ready_timeout_s: float,
        setup_navigation_timeout_s: float,
        per_tool_timeout_s: float,
        episode_timeout_s: float,
        max_repeated_tool_calls: int,
    ) -> None:
        # Keep a preflight-selected Lexmount region for every real rollout;
        # otherwise a successful regional smoke test and a training session
        # could silently use different egress paths.
        lexmount_kwargs: dict[str, str] = {
            "api_key": api_key,
            "project_id": project_id,
        }
        region = os.environ.get("LEXMOUNT_REGION", "").strip()
        if region:
            lexmount_kwargs["region"] = region
        self.lexmount = Lexmount(**lexmount_kwargs)
        if dom_backend not in {"stagehand", "cdp"}:
            raise ValueError(f"Unsupported dom_backend={dom_backend!r}")
        self.dom_backend = dom_backend
        self.stagehand_model = stagehand_model
        self.policy_model = policy_model
        self.proxy_model_to_stagehand = proxy_model_to_stagehand
        self.browser_mode = browser_mode
        self.official_proxy = official_proxy
        self.external_proxy = external_proxy
        self.stagehand_ready_timeout_s = stagehand_ready_timeout_s
        # This deadline is intentionally separate from a policy tool-call
        # deadline.  Initial navigation can require a cold real-site load and
        # CDP attachment, while a policy action must remain tightly bounded.
        self.setup_navigation_timeout_s = setup_navigation_timeout_s
        self.per_tool_timeout_s = per_tool_timeout_s
        self.episode_timeout_s = episode_timeout_s
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.stagehand_client: AsyncStagehand | None = None
        self._client_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_concurrent_sessions)
        self.session_create_timeout_s = session_create_timeout_s
        self._background_session_cleanup_tasks: set[asyncio.Task[None]] = set()
        self.logger = LOGGER

    def register_tools(self, env: vf.StatefulToolEnv) -> None:
        self.logger = env.logger
        env.add_tool(self.navigate, args_to_skip=["session", "guard"])
        env.add_tool(self.observe, args_to_skip=["session", "llm_config", "guard"])
        env.add_tool(self.act, args_to_skip=["session", "llm_config", "guard"])
        # ``extract`` requires a nested JSON-schema string.  That is useful
        # for Stagehand, but is neither used nor needed by the deterministic
        # CDP adapter: ``observe`` already returns literal rendered evidence.
        # Omitting it here keeps a small policy action space and prevents a
        # malformed nested schema from being misclassified as a tool-parser
        # failure.  Stagehand retains its native structured extractor.
        if self.dom_backend != "cdp":
            env.add_tool(self.extract, args_to_skip=["session", "llm_config", "guard"])

    async def _get_stagehand_client(self) -> AsyncStagehand:
        async with self._client_lock:
            if self.stagehand_client is None:
                self.stagehand_client = AsyncStagehand(
                    server="local",
                    browserbase_api_key="local-cdp-not-browserbase",
                    model_api_key=os.environ.get("MODEL_API_KEY") or os.environ["OPENAI_API_KEY"],
                    local_ready_timeout_s=self.stagehand_ready_timeout_s,
                    # Stagehand's default HTTP timeout can expire while a cold
                    # real site is still loading.  Keep this greater than the
                    # setup-only deadline below; policy tools are still bounded
                    # independently by asyncio.wait_for(per_tool_timeout_s).
                    timeout=self.setup_navigation_timeout_s,
                    max_retries=0,
                )
        return self.stagehand_client

    async def setup_state(self, state: vf.State) -> vf.State:
        guard = state.get("trajectory_guard")
        if not isinstance(guard, TrajectoryGuard):
            guard = TrajectoryGuard(
                per_tool_timeout_s=self.per_tool_timeout_s,
                episode_timeout_s=self.episode_timeout_s,
                max_repeated_tool_calls=self.max_repeated_tool_calls,
            )
            state["trajectory_guard"] = guard
        slot_started_at = time.monotonic()
        await self._slots.acquire()
        guard.record_timing(
            "browser_slot_wait_seconds", slot_started_at, phase="browser_slot_wait"
        )
        state["lexbrowser_slot_acquired"] = True
        lexmount_session = None
        try:
            session_kwargs: dict[str, Any] = {"browser_mode": self.browser_mode}
            if self.external_proxy is not None:
                # Lexmount's native authenticated upstream-proxy session
                # configuration.  Use it instead of (not in addition to) the
                # provider proxy so egress is deterministic per session.
                session_kwargs["proxy"] = self.external_proxy
            else:
                session_kwargs["official_proxy"] = self.official_proxy
            create_started_at = time.monotonic()
            create_task: asyncio.Task[Any] = asyncio.create_task(
                asyncio.to_thread(self.lexmount.sessions.create, **session_kwargs)
            )
            try:
                lexmount_session = await asyncio.wait_for(
                    asyncio.shield(create_task),
                    timeout=self.session_create_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                cleanup_task = asyncio.create_task(
                    self._cleanup_late_session_create(
                        create_task,
                        create_started_at=create_started_at,
                    )
                )
                self._background_session_cleanup_tasks.add(cleanup_task)
                cleanup_task.add_done_callback(
                    self._background_session_cleanup_tasks.discard
                )
                guard.record_timing(
                    "lexmount_session_create_seconds",
                    create_started_at,
                    phase="lexmount_session_create",
                    status="error:TimeoutError",
                )
                self.logger.warning(
                    "Lexmount session creation timed out after %.2fs; "
                    "late-created session will be closed if the provider returns it",
                    time.monotonic() - create_started_at,
                )
                raise exc
            except Exception as exc:
                guard.record_timing(
                    "lexmount_session_create_seconds",
                    create_started_at,
                    phase="lexmount_session_create",
                    status=f"error:{type(exc).__name__}",
                )
                self.logger.warning(
                    "Lexmount session creation failed after %.2fs: %r",
                    time.monotonic() - create_started_at,
                    exc,
                )
                raise
            else:
                guard.record_timing(
                    "lexmount_session_create_seconds",
                    create_started_at,
                    phase="lexmount_session_create",
                )
            cdp_url = lexmount_session.connect_url
            if not cdp_url:
                raise RuntimeError("Lexmount session did not return a CDP URL")
            state["lexmount_session"] = lexmount_session
            state["lexmount_session_id"] = lexmount_session.id
            browser_attach_started_at = time.monotonic()
            if self.dom_backend == "cdp":
                state["browser_session"] = await asyncio.to_thread(
                    LexmountCDPSession, cdp_url, self.setup_navigation_timeout_s
                )
            else:
                client = await self._get_stagehand_client()
                stagehand_session = await client.sessions.start(
                    model_name=self.stagehand_model,
                    browser={"type": "local", "cdp_url": cdp_url},
                    verbose=0,
                )
                state["browser_session"] = stagehand_session
                state["stagehand_session"] = stagehand_session
                state["stagehand_session_id"] = stagehand_session.id
            guard.record_timing(
                "browser_attach_seconds",
                browser_attach_started_at,
                phase="browser_attach",
            )
            return state
        except Exception:
            if lexmount_session is not None:
                await self._close_lexmount_session(
                    lexmount_session, reason="setup_state_exception"
                )
            self._release_slot(state)
            raise

    async def _cleanup_late_session_create(
        self,
        create_task: asyncio.Task[Any],
        *,
        create_started_at: float,
    ) -> None:
        try:
            lexmount_session = await create_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                "Late Lexmount session create finished with error after %.2fs: %r",
                time.monotonic() - create_started_at,
                exc,
            )
            return
        await self._close_lexmount_session(
            lexmount_session,
            reason=(
                "session_create_timeout_late_cleanup "
                f"after={time.monotonic() - create_started_at:.2f}s"
            ),
        )

    async def _close_lexmount_session(self, lexmount_session: Any, *, reason: str) -> None:
        session_id = str(
            getattr(lexmount_session, "id", None)
            or getattr(lexmount_session, "session_id", None)
            or ""
        )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(lexmount_session.close),
                timeout=30.0,
            )
            self.logger.info(
                "Closed Lexmount session%s (%s)",
                f" {session_id}" if session_id else "",
                reason,
            )
            return
        except Exception as exc:
            self.logger.warning(
                "Failed to close Lexmount session%s (%s): %r",
                f" {session_id}" if session_id else "",
                reason,
                exc,
            )
        if session_id:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self.lexmount.sessions.delete,
                        session_id=session_id,
                    ),
                    timeout=30.0,
                )
                self.logger.info(
                    "Deleted Lexmount session %s after close failure (%s)",
                    session_id,
                    reason,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to delete Lexmount session %s after close failure (%s): %r",
                    session_id,
                    reason,
                    exc,
                )

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
            updated["session"] = state["browser_session"]
            updated["guard"] = state["trajectory_guard"]
        if tool_name in {"observe", "act", "extract"}:
            updated["llm_config"] = self._llm_config(state)
        return updated

    def _release_slot(self, state: vf.State) -> None:
        if state.pop("lexbrowser_slot_acquired", False):
            self._slots.release()

    async def cleanup_session(self, state: vf.State) -> None:
        try:
            browser_session = state.pop("browser_session", None)
            stagehand_session = state.pop("stagehand_session", None)
            if stagehand_session is not None:
                try:
                    await stagehand_session.end()
                except Exception as exc:
                    self.logger.warning("Failed to end Stagehand session: %s", exc)
            elif isinstance(browser_session, LexmountCDPSession):
                await asyncio.to_thread(browser_session.close)
            lexmount_session = state.pop("lexmount_session", None)
            if lexmount_session is not None:
                await self._close_lexmount_session(
                    lexmount_session, reason="rollout_cleanup"
                )
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
        started_at = time.monotonic()
        try:
            if isinstance(session, LexmountCDPSession):
                await asyncio.wait_for(
                    asyncio.to_thread(session.navigate, url), timeout=guard.per_tool_timeout_s
                )
                # CDP ``Page.navigate`` acknowledges before the target page
                # is readable.  Do not hand the policy an empty interstitial
                # and let it hallucinate a final answer: wait for rendered DOM
                # evidence within this tool's bounded budget.
                await asyncio.wait_for(
                    asyncio.to_thread(
                        session.wait_for_usable_document,
                        max(0.1, guard.per_tool_timeout_s - 0.5),
                    ),
                    timeout=guard.per_tool_timeout_s,
                )
                return f"Navigated to {url}"
            await asyncio.wait_for(
                session.navigate(
                    url=url,
                    # Do not let a continuously busy real page turn a simple
                    # navigation into an unbounded network-idle wait.
                    options={
                        "wait_until": "domcontentloaded",
                        "timeout": int(guard.per_tool_timeout_s * 1000),
                    },
                    timeout=guard.per_tool_timeout_s + 5.0,
                ),
                timeout=guard.per_tool_timeout_s + 5.0,
            )
            return f"Navigated to {url}"
        except asyncio.TimeoutError:
            return f"ERROR_{guard.infrastructure_timeout('navigate').upper()}: retryable browser timeout"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_NAVIGATE: {exc}"
        finally:
            guard.record_timing(
                "browser_tool_seconds", started_at, phase="browser_navigate"
            )
            guard.record_timing("browser_navigate_seconds", started_at)
            guard.timings["browser_tool_count"] = guard.timings.get("browser_tool_count", 0.0) + 1.0

    async def observe(
        self, instruction: str, session: Any, llm_config: Any = None, guard: TrajectoryGuard | None = None
    ) -> str:
        """Find possible page actions matching an instruction."""
        assert guard is not None
        blocked = guard.before_tool("observe", instruction)
        if blocked:
            return f"ERROR_{blocked.upper()}: trajectory terminated"
        started_at = time.monotonic()
        try:
            if isinstance(session, LexmountCDPSession):
                actions = await asyncio.wait_for(
                    asyncio.to_thread(session.observe), timeout=guard.per_tool_timeout_s
                )
                return json.dumps(actions, ensure_ascii=False, separators=(",", ":")) if actions else "No matching elements found"
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
            return json.dumps(actions, ensure_ascii=False, separators=(",", ":")) if actions else "No matching elements found"
        except asyncio.TimeoutError:
            return f"ERROR_{guard.infrastructure_timeout('observe').upper()}: retryable browser timeout"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_OBSERVE: {exc}"
        finally:
            guard.record_timing(
                "browser_tool_seconds", started_at, phase="browser_observe"
            )
            guard.record_timing("browser_observe_seconds", started_at)
            guard.timings["browser_tool_count"] = guard.timings.get("browser_tool_count", 0.0) + 1.0

    async def act(
        self, instruction: str, session: Any, llm_config: Any = None, guard: TrajectoryGuard | None = None
    ) -> str:
        """Perform one natural-language action on the current page."""
        assert guard is not None
        blocked = guard.before_tool("act", instruction)
        if blocked:
            return f"ERROR_{blocked.upper()}: trajectory terminated"
        started_at = time.monotonic()
        try:
            if isinstance(session, LexmountCDPSession):
                result = await asyncio.wait_for(
                    asyncio.to_thread(session.act, instruction), timeout=guard.per_tool_timeout_s
                )
                if "clicked" in result:
                    evidence = await asyncio.wait_for(
                        asyncio.to_thread(session.visible_text), timeout=guard.per_tool_timeout_s
                    )
                    # Surface the post-click rendered page, not an inferred
                    # answer.  This gives both policy and transcript judge
                    # direct browser evidence even if the policy terminates
                    # immediately after a real form submission.
                    return (
                        "Success: clicked. Rendered page evidence: "
                        f"{evidence[-1800:]}. Next call observe for the full page; do not use extract."
                    )
                return f"Success: {result}"
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
        except ValueError as exc:
            # Invalid selector / unresolved observed-control instructions are
            # policy grounding mistakes, not a Lexmount or network outage.
            guard.policy_failures += 1
            return f"ERROR_POLICY_ACT: {exc}"
        except Exception as exc:
            guard.infrastructure_failures += 1
            return f"ERROR_INFRASTRUCTURE_ACT: {exc}"
        finally:
            guard.record_timing("browser_tool_seconds", started_at, phase="browser_act")
            guard.record_timing("browser_act_seconds", started_at)
            guard.timings["browser_tool_count"] = guard.timings.get("browser_tool_count", 0.0) + 1.0

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
        started_at = time.monotonic()
        try:
            schema = json.loads(schema_json)
            if isinstance(session, LexmountCDPSession):
                # Return rendered DOM evidence rather than using Stagehand's
                # LLM-based extractor.  Values are deliberately not parsed or
                # inferred: the policy/judge receives the literal page text.
                evidence = session.visible_text()
                return json.dumps(
                    {key: evidence for key in schema.get("properties", {})}, ensure_ascii=False
                )
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
        finally:
            guard.record_timing(
                "browser_tool_seconds", started_at, phase="browser_extract"
            )
            guard.record_timing("browser_extract_seconds", started_at)
            guard.timings["browser_tool_count"] = guard.timings.get("browser_tool_count", 0.0) + 1.0


class LexBrowserEnv(vf.StatefulToolEnv):
    def __init__(self, *, mode_impl: LexmountDOMMode, **kwargs: Any) -> None:
        super().__init__(stop_errors=[vf.SandboxError], **kwargs)
        self._mode_impl = mode_impl
        self._mode_impl.register_tools(self)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        try:
            state = await self._mode_impl.setup_state(state)
            info = state.get("info") or {}
            session = state.get("browser_session")
            if isinstance(session, LexmountCDPSession):
                session.set_task_query(str(info.get("question") or state.get("question") or ""))
            start_url = info.get("start_url") or state.get("start_url")
            if start_url:
            # BrowserEnv reset semantics: WebVoyager begins on the dataset's
            # website. This navigation is environment setup, not an agent tool call.
            # A real site can miss Stagehand's internal 15s DOM-ready deadline.
            # Treat that as transient browser infrastructure: discard the CDP
            # session and retry from a fresh Lexmount Chrome.  Retrying the same
            # Stagehand session is unsafe because a late navigation can supersede
            # the next request.
                last_error: Exception | None = None
                for attempt in range(1, SETUP_NAVIGATION_MAX_ATTEMPTS + 1):
                    navigation_started_at = time.monotonic()
                    try:
                        session = state["browser_session"]
                        attempt_deadline = time.monotonic() + self._mode_impl.setup_navigation_timeout_s
                        if isinstance(session, LexmountCDPSession):
                            await asyncio.wait_for(
                                asyncio.to_thread(session.navigate, start_url),
                                timeout=self._mode_impl.setup_navigation_timeout_s,
                            )
                            remaining = max(0.1, attempt_deadline - time.monotonic())
                            await asyncio.wait_for(
                                asyncio.to_thread(
                                    session.wait_for_usable_document,
                                    remaining,
                                ),
                                timeout=remaining + 1.0,
                            )
                        else:
                            await asyncio.wait_for(
                                session.navigate(
                                    url=start_url,
                                    # This must be a Stagehand navigation option (ms),
                                    # not only an outer Python timeout.  The former
                                    # controls the server-side page wait strategy.
                                    options={
                                        "wait_until": "domcontentloaded",
                                        "timeout": int(
                                            self._mode_impl.setup_navigation_timeout_s * 1000
                                        ),
                                    },
                                    timeout=self._mode_impl.setup_navigation_timeout_s + 5.0,
                                ),
                                timeout=self._mode_impl.setup_navigation_timeout_s + 10.0,
                            )
                        state["setup_navigation_attempts"] = attempt
                        state["setup_navigation_retry_success"] = float(attempt > 1)
                        guard = state.get("trajectory_guard")
                        if isinstance(guard, TrajectoryGuard):
                            guard.record_timing(
                                "browser_setup_navigation_seconds",
                                navigation_started_at,
                                phase="browser_setup_navigation",
                            )
                        break
                    except Exception as exc:
                        last_error = exc
                        guard = state.get("trajectory_guard")
                        if guard is not None:
                            guard.infrastructure_failures += 1
                        state["setup_navigation_attempts"] = attempt
                        if isinstance(guard, TrajectoryGuard):
                            guard.record_timing(
                                "browser_setup_navigation_seconds",
                                navigation_started_at,
                                phase="browser_setup_navigation",
                                status=f"error:{type(exc).__name__}",
                            )
                        self._mode_impl.logger.warning(
                            "Initial navigation failed (attempt %s/%s, url=%s): %r",
                            attempt,
                            SETUP_NAVIGATION_MAX_ATTEMPTS,
                            start_url,
                            exc,
                        )
                        await self._mode_impl.cleanup_session(state)
                        if attempt == SETUP_NAVIGATION_MAX_ATTEMPTS:
                            raise RuntimeError(
                                "infrastructure_setup_navigation_failed"
                            ) from exc
                        await asyncio.sleep(0.5 * attempt)
                        state = await self._mode_impl.setup_state(state)
                if last_error is not None and "setup_navigation_retry_success" not in state:
                    raise RuntimeError("infrastructure_setup_navigation_failed") from last_error
                state["initial_url"] = start_url
            initialized = await super().setup_state(state, **kwargs)
            return initialized if initialized is not None else state
        except BaseException:
            # A caller cancellation can occur while setup_state owns a
            # Lexmount session slot.  StatefulToolEnv only invokes cleanup
            # after a normally completed rollout, so release the project
            # resources here before propagating the cancellation/error.
            await self._mode_impl.cleanup_session(state)
            raise

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


def _external_proxy_from_env() -> dict[str, str] | None:
    """Read an optional Lexmount SDK external proxy without leaking secrets.

    The absent default keeps the BrowserEnv-aligned provider egress.  If an
    operator supplies a proxy server, require a complete authenticated tuple
    rather than launching a partly configured session that would fail during
    real-site navigation.
    """
    server = os.environ.get("LEXMOUNT_EXTERNAL_PROXY_SERVER", "").strip()
    if not server:
        return None
    username = os.environ.get("LEXMOUNT_EXTERNAL_PROXY_USERNAME", "").strip()
    password = os.environ.get("LEXMOUNT_EXTERNAL_PROXY_PASSWORD", "").strip()
    if not username or not password:
        raise ValueError(
            "LEXMOUNT_EXTERNAL_PROXY_SERVER requires both "
            "LEXMOUNT_EXTERNAL_PROXY_USERNAME and LEXMOUNT_EXTERNAL_PROXY_PASSWORD"
        )
    return {
        "type": "external",
        "server": server,
        "username": username,
        "password": password,
    }


def load_environment(
    mode: str = "dom",
    dom_backend: str = "stagehand",
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
    session_create_timeout_s: float = 60.0,
    stagehand_ready_timeout_s: float = 60.0,
    setup_navigation_timeout_s: float = 90.0,
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
        # GLM-5.2 is a reasoning model.  With the upstream eight-token cap it
        # consumes the entire budget in ``reasoning_content`` and returns an
        # empty assistant message, which forces every binary reward to zero.
        # 1024 is verified to reach a literal yes/no completion; this is
        # scorer-only and does not affect policy generation or packing.
        judge_sampling_args={"temperature": 0.0, "max_tokens": 1024},
    )
    rubric.add_reward_func(judge_task_completion, weight=1.0)

    mode_impl = LexmountDOMMode(
        api_key=_required_env("LEXMOUNT_API_KEY"),
        project_id=_required_env("LEXMOUNT_PROJECT_ID"),
        dom_backend=dom_backend,
        stagehand_model=stagehand_model,
        policy_model=policy_model,
        proxy_model_to_stagehand=proxy_model_to_stagehand,
        browser_mode=browser_mode,
        official_proxy=official_proxy,
        external_proxy=_external_proxy_from_env(),
        max_concurrent_sessions=max_concurrent_sessions,
        session_create_timeout_s=session_create_timeout_s,
        stagehand_ready_timeout_s=stagehand_ready_timeout_s,
        setup_navigation_timeout_s=setup_navigation_timeout_s,
        per_tool_timeout_s=per_tool_timeout_s,
        episode_timeout_s=episode_timeout_s,
        max_repeated_tool_calls=max_repeated_tool_calls,
    )
    return LexBrowserEnv(
        mode_impl=mode_impl,
        dataset=dataset,
        system_prompt=(
            CDP_BROWSER_AGENT_SYSTEM_PROMPT if dom_backend == "cdp" else BROWSER_AGENT_SYSTEM_PROMPT
        ),
        rubric=rubric,
        parser=parser,
        max_turns=max_turns,
        **kwargs,
    )
