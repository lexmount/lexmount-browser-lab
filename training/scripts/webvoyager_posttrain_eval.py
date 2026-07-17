#!/usr/bin/env python3
# ruff: noqa: E501, UP017
"""Run a training-protocol-compatible WebVoyager checkpoint evaluation.

The 910B WebVoyager GRPO run trained Qwen3-8B against one function named
``browser(operation, instruction)``.  This utility deliberately preserves
that policy-facing contract and swaps only the CDP session implementation:
Lexmount Chrome or a fresh local Chrome profile.

Use ``prepare-splits`` once to create stable paired task manifests, then run
the same manifest once per backend.  Raw trajectories remain local runner
artifacts; the JSONL output is resumable and contains no credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import re
import socket
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINING_SYSTEM_PROMPT = (
    "You are an autonomous browser agent. A real browser is open on the task website. "
    "First call browser with operation=observe. Use the returned [data-lex-id=lex-N] "
    "selectors for grounded actions. Call browser with operation=act and an instruction "
    "such as `fill [data-lex-id=lex-0] :: text` or `click [data-lex-id=lex-1]`. "
    "Use one browser action per turn, inspect the resulting page, and only then provide "
    "a concise final answer supported by browser evidence."
)

BROWSER_TOOL = {
    "type": "function",
    "function": {
        "name": "browser",
        "description": "Use the real browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "One of observe, act, or navigate.",
                },
                "instruction": {
                    "type": "string",
                    "description": "For act use a grounded selector action; for navigate use a URL.",
                },
            },
            "required": ["operation"],
        },
    },
}

TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT = """Judge whether the browser agent completed the task.

The input contains the task, optional rubric, initial environment state, execution status, complete recorded action/tool-result trajectory, final URL, final DOM/accessibility snapshot, optional screenshot evidence, and the policy's final response. Use the available evidence and be reasonably permissive when it supports completion.

Return exactly one JSON object with no Markdown and no additional keys:
{{"reason":"a short evidence-based explanation","verdict":"yes|no"}}

Task:
```
{question}
```

Rubric:
```
{rubric}
```

Initial Environment State:
```
{initial_state}
```

Execution Status:
```json
{execution_status}
```

Action and Tool-Result Trajectory:
```
{response}
```

Final URL:
```
{final_url}
```

Final DOM / Accessibility Snapshot:
```
{final_state}
```

Final Screenshot / Key Screenshots:
```
{screenshot_evidence}
```

Policy Final Response:
```
{final_answer}
```"""

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
GIB = 1024**3
MIN_USABLE_VISIBLE_TEXT_CHARS = 160
MIN_USABLE_ACTIONABLE_ELEMENTS = 2


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    start_url: str
    website: str
    split: str = "unspecified"

    def as_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "start_url": self.start_url,
            "website": self.website,
            "split": self.split,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_env_file(path: Path) -> None:
    """Load simple dotenv assignments without echoing or evaluating their values."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier() or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def task_from_mapping(row: Mapping[str, Any], *, split: str = "unspecified") -> Task:
    task_id = str(row.get("task_id") or row.get("id") or "").strip()
    question = str(row.get("question") or row.get("ques") or "").strip()
    start_url = str(row.get("start_url") or row.get("web") or "").strip()
    website = str(row.get("website") or row.get("web_name") or "unknown").strip() or "unknown"
    resolved_split = str(row.get("split") or split).strip() or split
    if not task_id or not question or not start_url:
        raise ValueError(f"task record is missing task_id/question/start_url: {dict(row)!r}")
    return Task(task_id, question, start_url, website, resolved_split)


def load_jsonl_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"task record at {path}:{line_number} must be an object")
        tasks.append(task_from_mapping(row))
    ensure_unique_task_ids(tasks, path)
    return tasks


def load_training_parquet(path: Path) -> list[Task]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("prepare-splits requires pyarrow in the active Python environment") from exc

    rows = parquet.read_table(path).to_pylist()
    tasks: list[Task] = []
    for index, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        tools_kwargs = extra_info.get("tools_kwargs") or {}
        browser_kwargs = tools_kwargs.get("browser") or {}
        create_kwargs = browser_kwargs.get("create_kwargs") or {}
        question = str(create_kwargs.get("question") or "").strip()
        start_url = str(create_kwargs.get("start_url") or "").strip()
        task_id = str(create_kwargs.get("task_id") or "").strip()
        if not question and isinstance(row.get("prompt"), list):
            for message in row["prompt"]:
                if isinstance(message, dict) and message.get("role") == "user":
                    question = str(message.get("content") or "").strip()
                    break
        if not task_id or not question or not start_url:
            raise ValueError(f"unusable training record at parquet row {index}")
        tasks.append(Task(task_id, question, start_url, task_id.split("--", 1)[0], "train"))
    ensure_unique_task_ids(tasks, path)
    return tasks


def ensure_unique_task_ids(tasks: Sequence[Task], source: Path) -> None:
    counts = Counter(task.task_id for task in tasks)
    duplicates = sorted(task_id for task_id, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate task ids in {source}: {duplicates[:8]}")


def write_task_jsonl(path: Path, tasks: Iterable[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.as_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")


def diverse_sample(tasks: Sequence[Task], count: int, seed: int) -> list[Task]:
    if count > len(tasks):
        raise ValueError(f"requested {count} tasks from a population of {len(tasks)}")
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[task.website].append(task)
    rng = random.Random(seed)
    ordered_groups: list[list[Task]] = []
    for website in sorted(groups):
        group = list(groups[website])
        rng.shuffle(group)
        ordered_groups.append(group)

    selected: list[Task] = []
    cursor = 0
    while len(selected) < count:
        progress = False
        for group in ordered_groups:
            if cursor < len(group):
                selected.append(group[cursor])
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise RuntimeError("diverse task selection exhausted unexpectedly")
        cursor += 1
    return selected


def prepare_splits(args: argparse.Namespace) -> int:
    training_parquet = args.training_parquet.resolve()
    benchmark_jsonl = args.benchmark_jsonl.resolve()
    output_dir = args.output_dir.resolve()
    training_tasks = load_training_parquet(training_parquet)
    benchmark_tasks = load_jsonl_tasks(benchmark_jsonl)
    benchmark_by_id = {task.task_id: task for task in benchmark_tasks}
    train_ids = {task.task_id for task in training_tasks}
    in_train = [
        Task(task.task_id, task.question, task.start_url, task.website, "in_train")
        for task in benchmark_tasks
        if task.task_id in train_ids
    ]
    holdout = [
        Task(task.task_id, task.question, task.start_url, task.website, "holdout")
        for task in benchmark_tasks
        if task.task_id not in train_ids
    ]
    missing = train_ids - set(benchmark_by_id)
    if missing:
        raise RuntimeError(f"{len(missing)} training task ids are missing from benchmark JSONL")
    if len(in_train) != len(training_tasks):
        raise RuntimeError("in-train overlap count does not match training parquet")

    smoke = diverse_sample(in_train, args.smoke_count, args.selection_seed)
    train_sample = diverse_sample(in_train, args.train_count, args.selection_seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_task_jsonl(output_dir / f"smoke_{len(smoke)}.jsonl", smoke)
    write_task_jsonl(output_dir / f"in_train_{len(train_sample)}.jsonl", train_sample)
    write_task_jsonl(output_dir / f"holdout_{len(holdout)}.jsonl", holdout)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol": "webvoyager-posttrain-v1",
        "selection_seed": args.selection_seed,
        "inputs": {
            "training_parquet": str(training_parquet),
            "training_parquet_sha256": sha256_file(training_parquet),
            "benchmark_jsonl": str(benchmark_jsonl),
            "benchmark_jsonl_sha256": sha256_file(benchmark_jsonl),
        },
        "counts": {
            "training_parquet": len(training_tasks),
            "benchmark": len(benchmark_tasks),
            "in_train": len(in_train),
            "holdout": len(holdout),
            "smoke": len(smoke),
            "in_train_sample": len(train_sample),
        },
        "outputs": {
            "smoke": str(output_dir / f"smoke_{len(smoke)}.jsonl"),
            "in_train": str(output_dir / f"in_train_{len(train_sample)}.jsonl"),
            "holdout": str(output_dir / f"holdout_{len(holdout)}.jsonl"),
        },
        "websites": {
            "in_train": dict(sorted(Counter(task.website for task in in_train).items())),
            "holdout": dict(sorted(Counter(task.website for task in holdout).items())),
            "smoke": dict(sorted(Counter(task.website for task in smoke).items())),
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def openai_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def _tool_call_dicts(message: Any) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        name = str(getattr(function, "name", "") or "")
        arguments = str(getattr(function, "arguments", "{}") or "{}")
        call_id = str(getattr(call, "id", "") or f"call_{len(calls)}")
        if name:
            calls.append({"id": call_id, "name": name, "arguments": arguments})
    if calls:
        return calls
    for text in (getattr(message, "content", None), getattr(message, "reasoning_content", None)):
        if not isinstance(text, str):
            continue
        for match in TOOL_CALL_PATTERN.finditer(text):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not payload.get("name"):
                continue
            arguments = payload.get("arguments", {})
            calls.append(
                {
                    "id": f"call_recovered_{len(calls)}",
                    "name": str(payload["name"]),
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                }
            )
    return calls


def _visible_final_answer(content: str, generation_truncated: bool) -> tuple[str, str]:
    if generation_truncated:
        return "", "generation_truncated"
    text = content.strip()
    if re.search(r"<think\b", text, flags=re.IGNORECASE):
        text = THINK_PATTERN.sub("", text).strip()
        if re.search(r"</?think\b", text, flags=re.IGNORECASE):
            return "", "no_final_answer"
    text = re.sub(r"^<final>\s*|\s*</final>$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<\|(?:im_start|im_end|endoftext)\|>", "", text).strip()
    return (text, "complete") if text else ("", "no_final_answer")


def _truncate_middle(text: str, limit: int = 60_000) -> tuple[str, bool]:
    marker = "...(truncated)..."
    if len(text) <= limit:
        return text, False
    if limit <= len(marker):
        return text[:limit], True
    budget = limit - len(marker)
    left = budget // 2
    return text[:left] + marker + text[-(budget - left) :], True


def render_transcript(events: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    if not events:
        return "", False
    call_lines = [
        "TOOL_CALL browser(" + json.dumps(event["parameters"], ensure_ascii=False) + ")"
        for event in events
    ]
    fixed = sum(len(line) + len("\nTOOL_RESULT: \n") for line in call_lines)
    result_budget = max(512, (60_000 - fixed) // len(events))
    chunks: list[str] = []
    truncated = False
    for line, event in zip(call_lines, events, strict=True):
        result, shortened = _truncate_middle(str(event["result"]), result_budget)
        chunks.append(f"{line}\nTOOL_RESULT: {result}")
        truncated = truncated or shortened
    transcript, shortened = _truncate_middle("\n".join(chunks))
    return transcript, truncated or shortened


def _error_classification(text: str) -> str | None:
    normalized = text.lstrip().upper()
    if normalized.startswith("ERROR_INFRASTRUCTURE") or normalized.startswith("ERROR_ENVIRONMENT"):
        return "infrastructure"
    if normalized.startswith("ERROR_POLICY"):
        return "policy"
    return None


async def _final_snapshot(session: Any) -> tuple[str, str]:
    try:
        snapshot = await asyncio.wait_for(asyncio.to_thread(session.observe), timeout=25.0)
    except Exception as exc:
        return "", f"Final snapshot unavailable: {type(exc).__name__}"
    if not isinstance(snapshot, dict):
        return "", "Final snapshot unavailable: unsupported snapshot result"
    return str(snapshot.get("url") or ""), json.dumps(snapshot, ensure_ascii=False)


async def _judge_task(
    client: Any,
    *,
    model: str,
    task: Task,
    transcript: str,
    final_answer: str,
    execution_status: Mapping[str, Any],
    final_url: str,
    final_state: str,
) -> dict[str, Any]:
    prompt = TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT.format(
        question=task.question,
        rubric="Not provided for this task.",
        initial_state=json.dumps({"start_url": task.start_url}, ensure_ascii=False),
        execution_status=json.dumps(execution_status, ensure_ascii=False, sort_keys=True),
        response=transcript,
        final_url=final_url or "Unavailable",
        final_state=final_state,
        screenshot_evidence="Unavailable: this run uses a text-only Judge.",
        final_answer=final_answer,
    )
    started = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = str(response.choices[0].message.content or "").strip()
        payload = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE))
        if not isinstance(payload, dict) or payload.get("verdict") not in {"yes", "no"}:
            raise ValueError("judge response did not contain verdict=yes|no")
        return {
            "status": "ok",
            "reward": 1.0 if payload["verdict"] == "yes" else 0.0,
            "verdict": payload["verdict"],
            "reason": str(payload.get("reason") or ""),
            "latency_seconds": round(time.monotonic() - started, 4),
        }
    except Exception as exc:
        return {
            "status": "error",
            "reward": None,
            "verdict": None,
            "reason": f"{type(exc).__name__}: {exc}",
            "latency_seconds": round(time.monotonic() - started, 4),
        }


async def _open_browser_state(mode: Any, task: Task, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    last_error: BaseException | None = None
    for attempt in range(1, args.setup_attempts + 1):
        state: dict[str, Any] = {"info": {"question": task.question, "start_url": task.start_url}}
        try:
            state = await mode.setup_state(state)
            session = state["browser_session"]
            if callable(getattr(session, "set_task_query", None)):
                session.set_task_query(task.question)
            result = await mode.navigate(task.start_url, session, state["trajectory_guard"])
            if _error_classification(result):
                raise RuntimeError(result)
            state["setup_navigation_result"] = result
            state["setup_attempts"] = attempt
            return state, attempt
        except BaseException as exc:
            last_error = exc
            await mode.cleanup_session(state)
            if attempt < args.setup_attempts:
                await asyncio.sleep(0.5 * attempt)
    assert last_error is not None
    raise RuntimeError(f"browser setup failed after {args.setup_attempts} attempts: {last_error}") from last_error


async def evaluate_task(
    *,
    task: Task,
    ordinal: int,
    mode: Any,
    policy_client: Any,
    judge_client: Any | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_wall = utc_now()
    started = time.monotonic()
    seed = args.seed + ordinal
    events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": task.question},
    ]
    state: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": task.as_dict(),
        "backend": args.backend,
        "policy": {"model": args.model, "seed": seed},
        "started_at": started_wall,
        "status": "unknown",
        "events": events,
    }
    final_answer = ""
    final_answer_status = "no_final_answer"
    generation_truncated = False
    setup_attempts = 0
    try:
        state, setup_attempts = await _open_browser_state(mode, task, args)
        guard = state["trajectory_guard"]
        result["setup"] = {
            "attempts": setup_attempts,
            "navigation_result": state.get("setup_navigation_result", ""),
        }
        for turn in range(1, args.max_assistant_turns + 1):
            request_started = time.monotonic()
            completion = await policy_client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=[BROWSER_TOOL],
                tool_choice="auto",
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=seed,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            choice = completion.choices[0]
            message = choice.message
            model_seconds = time.monotonic() - request_started
            calls = _tool_call_dicts(message)
            raw_content = str(getattr(message, "content", "") or "")
            reasoning = str(getattr(message, "reasoning_content", "") or "")
            turn_record: dict[str, Any] = {
                "turn": turn,
                "finish_reason": str(choice.finish_reason or ""),
                "model_seconds": round(model_seconds, 4),
                "tool_call_count": len(calls),
            }
            result.setdefault("turns", []).append(turn_record)
            generation_truncated = generation_truncated or choice.finish_reason == "length"
            if not calls:
                final_answer, final_answer_status = _visible_final_answer(
                    raw_content or reasoning, generation_truncated
                )
                break
            if len(calls) != 1:
                final_answer_status = "policy_parallel_tool_calls"
                result["policy_protocol_error"] = "expected exactly one browser tool call per turn"
                break
            call = calls[0]
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": raw_content,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                ],
            }
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            messages.append(assistant_message)
            if call["name"] != "browser":
                tool_result = f"ERROR_POLICY_TOOL: unsupported tool {call['name']}"
                parameters = {"operation": "", "instruction": ""}
            else:
                try:
                    parameters = json.loads(call["arguments"])
                except json.JSONDecodeError:
                    parameters = {}
                    tool_result = "ERROR_POLICY_TOOL: browser arguments are not valid JSON"
                else:
                    operation = str(parameters.get("operation") or "")
                    instruction = str(parameters.get("instruction") or "")
                    tool_started = time.monotonic()
                    if operation == "observe":
                        tool_result = await mode.observe(
                            instruction, state["browser_session"], guard=guard
                        )
                    elif operation == "act":
                        tool_result = await mode.act(
                            instruction, state["browser_session"], guard=guard
                        )
                    elif operation == "navigate":
                        tool_result = await mode.navigate(
                            instruction, state["browser_session"], guard
                        )
                    else:
                        tool_result = "ERROR_POLICY_TOOL: operation must be observe, act, or navigate"
                    turn_record["browser_seconds"] = round(time.monotonic() - tool_started, 4)
            event = {
                "parameters": {
                    "operation": str(parameters.get("operation") or ""),
                    "instruction": str(parameters.get("instruction") or ""),
                },
                "result": str(tool_result),
            }
            events.append(event)
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": str(tool_result)}
            )
            if _error_classification(str(tool_result)) or bool(getattr(guard, "terminated", False)):
                final_answer_status = str(getattr(guard, "termination_reason", "")) or "tool_error"
                break
        else:
            final_answer_status = "max_assistant_turns"

        final_url, final_state = await _final_snapshot(state["browser_session"])
        transcript, transcript_truncated = render_transcript(events)
        execution_status = {
            "tool_call_count": len(events),
            "session_created": True,
            "episode_timed_out": bool(getattr(guard, "terminated", False))
            and "timeout" in str(getattr(guard, "termination_reason", "")),
            "generation_truncated": generation_truncated,
            "final_answer_present": bool(final_answer),
            "final_answer_status": final_answer_status,
            "transcript_truncated": transcript_truncated,
            "browser_error": "",
        }
        if judge_client is None:
            judge = {"status": "disabled", "reward": None, "verdict": None, "reason": ""}
        elif not events:
            judge = {"status": "skipped", "reward": 0.0, "verdict": "no", "reason": "no_tool_calls"}
        else:
            judge = await _judge_task(
                judge_client,
                model=args.judge_model,
                task=task,
                transcript=transcript,
                final_answer=final_answer,
                execution_status=execution_status,
                final_url=final_url,
                final_state=final_state,
            )
        result.update(
            {
                "status": "completed",
                "final_answer": final_answer,
                "final_answer_status": final_answer_status,
                "generation_truncated": generation_truncated,
                "guard": {
                    "termination_reason": str(getattr(guard, "termination_reason", "")),
                    "policy_failures": int(getattr(guard, "policy_failures", 0)),
                    "infrastructure_failures": int(getattr(guard, "infrastructure_failures", 0)),
                    "timeouts": int(getattr(guard, "timeouts", 0)),
                    "timings": dict(getattr(guard, "timings", {})),
                },
                "final_url": final_url,
                "final_state": final_state,
                "transcript": transcript,
                "judge": judge,
            }
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        setup_match = re.search(r"after (\d+) attempts", error)
        error_class = (
            "infrastructure"
            if "browser setup failed" in error or "ERROR_INFRASTRUCTURE" in error
            else "runner"
        )
        result.update(
            {
                "status": "setup_or_runner_error",
                "error_class": error_class,
                "error": error,
                "setup_attempts": int(setup_match.group(1)) if setup_match else setup_attempts,
                "final_answer": final_answer,
                "final_answer_status": final_answer_status,
            }
        )
    finally:
        if state is not None:
            try:
                await mode.cleanup_session(state)
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["finished_at"] = utc_now()
        result["wall_seconds"] = round(time.monotonic() - started, 4)
    return result


async def probe_task(
    *, task: Task, mode: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Measure one browser's usable-DOM availability without invoking a policy."""
    started = time.monotonic()
    state: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": task.as_dict(),
        "backend": args.backend,
        "status": "unknown",
    }
    try:
        state, attempts = await _open_browser_state(mode, task, args)
        guard = state["trajectory_guard"]
        observed = await mode.observe("", state["browser_session"], guard=guard)
        error_class = _error_classification(str(observed))
        result["setup"] = {
            "attempts": attempts,
            "navigation_result": state.get("setup_navigation_result", ""),
        }
        if error_class:
            result.update(
                {
                    "status": "browser_error",
                    "error_class": error_class,
                    "error": str(observed),
                }
            )
        else:
            snapshot = json.loads(str(observed))
            if not isinstance(snapshot, dict):
                raise ValueError("observe did not return a JSON object")
            visible_text = str(snapshot.get("visible_text") or "")
            elements = snapshot.get("elements")
            evidence = {
                "url": str(snapshot.get("url") or ""),
                "visible_text": visible_text,
                "element_count": len(elements) if isinstance(elements, list) else 0,
            }
            document = {
                "url": evidence["url"],
                "visible_text_chars": len(visible_text),
                "element_count": evidence["element_count"],
                "fingerprint_sha256": hashlib.sha256(
                    json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
            usable = (
                document["visible_text_chars"] >= MIN_USABLE_VISIBLE_TEXT_CHARS
                and document["element_count"] >= MIN_USABLE_ACTIONABLE_ELEMENTS
            )
            result.update(
                {
                    "status": "available" if usable else "degraded_document",
                    "document": document,
                }
            )
        result["guard"] = {
            "termination_reason": str(getattr(guard, "termination_reason", "")),
            "policy_failures": int(getattr(guard, "policy_failures", 0)),
            "infrastructure_failures": int(getattr(guard, "infrastructure_failures", 0)),
            "timeouts": int(getattr(guard, "timeouts", 0)),
            "timings": dict(getattr(guard, "timings", {})),
        }
    except Exception as exc:
        result.update(
            {
                "status": "setup_or_runner_error",
                "error_class": "infrastructure",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if state is not None:
            try:
                await mode.cleanup_session(state)
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["finished_at"] = utc_now()
        result["wall_seconds"] = round(time.monotonic() - started, 4)
    return result


def summarize_probe_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status") or "unknown") for row in rows)
    available = [
        row
        for row in rows
        if row.get("status") == "available"
        and int((row.get("document") or {}).get("visible_text_chars") or 0)
        >= MIN_USABLE_VISIBLE_TEXT_CHARS
        and int((row.get("document") or {}).get("element_count") or 0)
        >= MIN_USABLE_ACTIONABLE_ELEMENTS
    ]
    attempts = [
        int((row.get("setup") or {}).get("attempts") or 0)
        for row in rows
        if row.get("setup")
    ]
    wall = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds") is not None]
    return {
        "tasks": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "availability_rate": len(available) / len(rows) if rows else None,
        "usable_document_threshold": {
            "visible_text_chars": MIN_USABLE_VISIBLE_TEXT_CHARS,
            "actionable_elements": MIN_USABLE_ACTIONABLE_ELEMENTS,
        },
        "setup_attempts": {
            "mean": round(statistics.fmean(attempts), 4) if attempts else None,
            "max": max(attempts) if attempts else None,
        },
        "wall_seconds": {
            "mean": round(statistics.fmean(wall), 4) if wall else None,
            "max": round(max(wall), 4) if wall else None,
        },
    }


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status") or "unknown") for row in rows)
    final_answer_statuses = Counter(
        str(row.get("final_answer_status") or "unknown") for row in rows
    )
    judge_rows = [row.get("judge") for row in rows if isinstance(row.get("judge"), dict)]
    judged = [row for row in judge_rows if row.get("reward") is not None]
    rewards = [float(row["reward"]) for row in judged]
    errors = Counter()
    policy_failure_episodes = 0
    infrastructure_failure_episodes = 0
    timeout_episodes = 0
    for row in rows:
        if row.get("status") != "completed":
            errors[str(row.get("error_class") or "runner")] += 1
        guard = row.get("guard") or {}
        if guard.get("infrastructure_failures"):
            errors["infrastructure_episode"] += 1
            infrastructure_failure_episodes += 1
        if guard.get("policy_failures"):
            errors["policy_episode"] += 1
            policy_failure_episodes += 1
        if guard.get("timeouts"):
            errors["timeout_episode"] += 1
            timeout_episodes += 1
    wall = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds") is not None]
    return {
        "tasks": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "final_answer_statuses": dict(sorted(final_answer_statuses.items())),
        "trajectory": {
            "policy_failure_episodes": policy_failure_episodes,
            "infrastructure_failure_episodes": infrastructure_failure_episodes,
            "timeout_episodes": timeout_episodes,
        },
        "judge": {
            "judged": len(judged),
            "successes": int(sum(rewards)),
            "success_rate": (sum(rewards) / len(rewards)) if rewards else None,
            "errors": sum(1 for row in judge_rows if row.get("status") == "error"),
        },
        "error_categories": dict(sorted(errors.items())),
        "wall_seconds": {
            "mean": round(statistics.fmean(wall), 4) if wall else None,
            "max": round(max(wall), 4) if wall else None,
        },
    }


async def run_evaluation(args: argparse.Namespace) -> int:
    if args.env_file:
        parse_env_file(args.env_file.resolve())
    from lexbrowser_webvoyager_no_anti_bot.environment import LexmountDOMMode
    from openai import AsyncOpenAI

    tasks = load_jsonl_tasks(args.tasks.resolve())
    if args.task_id:
        requested_ids = set(args.task_id)
        tasks = [task for task in tasks if task.task_id in requested_ids]
        found_ids = {task.task_id for task in tasks}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise RuntimeError(f"requested task ids are absent from manifest: {missing_ids}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("selected task manifest is empty")
    if args.model_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.model_sha256):
        raise ValueError("--model-sha256 must be a 64-character hexadecimal SHA-256")
    model_artifact = args.model_artifact.resolve() if args.model_artifact else None
    if model_artifact is not None and not model_artifact.exists():
        raise FileNotFoundError(f"model artifact directory does not exist: {model_artifact}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    prior_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            prior_rows.append(row)
            if row.get("backend") == args.backend:
                task = row.get("task") or {}
                if row.get("status") == "completed" and task.get("task_id"):
                    completed_ids.add(str(task["task_id"]))

    policy_base_url = args.policy_base_url or _first_env("POLICY_BASE_URL", "QWEN_BASE_URL")
    policy_api_key = args.policy_api_key or _first_env("POLICY_API_KEY", "QWEN_API_KEY")
    if not policy_base_url or not policy_api_key:
        raise RuntimeError("set POLICY_BASE_URL/POLICY_API_KEY or QWEN_BASE_URL/QWEN_API_KEY")
    policy_base_url = openai_base_url(policy_base_url)
    policy_client = AsyncOpenAI(api_key=policy_api_key, base_url=policy_base_url)

    judge_client = None
    if args.judge == "training":
        judge_api_key = args.judge_api_key or _first_env("JUDGE_API_KEY", "OPENAI_API_KEY")
        judge_base_url = args.judge_base_url or _first_env("JUDGE_BASE_URL", "OPENAI_BASE_URL")
        if not judge_api_key or not judge_base_url:
            raise RuntimeError("training judge requires JUDGE_API_KEY/JUDGE_BASE_URL or OPENAI equivalents")
        args.judge_model = args.judge_model or _first_env("JUDGE_MODEL", "OPENAI_MODEL") or "glm-5.2"
        judge_client = AsyncOpenAI(api_key=judge_api_key, base_url=openai_base_url(judge_base_url))

    if args.backend == "lexmount":
        api_key = _required_env("LEXMOUNT_API_KEY")
        project_id = _required_env("LEXMOUNT_PROJECT_ID")
    else:
        api_key = None
        project_id = None
    mode = LexmountDOMMode(
        api_key=api_key,
        project_id=project_id,
        browser_backend=args.backend,
        dom_backend="cdp",
        stagehand_model="",
        policy_model="",
        proxy_model_to_stagehand=False,
        browser_mode="normal",
        official_proxy=args.lexmount_official_proxy,
        external_proxy=None,
        local_chrome_executable_path=args.local_chrome_executable,
        local_chrome_headless=not args.local_chrome_headed,
        local_proxy_server=args.local_proxy_server,
        local_proxy_bypass=args.local_proxy_bypass,
        max_concurrent_sessions=1,
        session_create_timeout_s=args.session_create_timeout,
        stagehand_ready_timeout_s=30.0,
        setup_navigation_timeout_s=args.setup_navigation_timeout,
        per_tool_timeout_s=args.per_tool_timeout,
        episode_timeout_s=args.episode_timeout,
        max_repeated_tool_calls=3,
    )
    manifest = {
        "schema_version": 1,
        "protocol": "webvoyager-posttrain-v1",
        "started_at": utc_now(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "backend": args.backend,
        "tasks": str(args.tasks.resolve()),
        "tasks_sha256": sha256_file(args.tasks.resolve()),
        "selected_tasks": len(tasks),
        "model": {
            "id": args.model,
            "base_url": policy_base_url.rstrip("/"),
            "artifact_dir": str(model_artifact) if model_artifact else None,
            "safetensors_sha256": args.model_sha256.lower() if args.model_sha256 else None,
        },
        "evaluator": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "repository_revision": repository_revision(),
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "enable_thinking": True,
            "seed_base": args.seed,
        },
        "browser": {
            "protocol": "browser(operation, instruction)",
            "dom_backend": "cdp",
            "max_assistant_turns": args.max_assistant_turns,
            "setup_attempts": args.setup_attempts,
            "local_chrome_headless": not args.local_chrome_headed,
            "local_proxy_configured": bool(args.local_proxy_server),
            "lexmount_official_proxy": args.lexmount_official_proxy,
        },
        "judge": {"mode": args.judge, "model": args.judge_model if judge_client else None},
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    rows: list[dict[str, Any]] = list(prior_rows)
    try:
        for ordinal, task in enumerate(tasks):
            if task.task_id in completed_ids:
                continue
            row = await evaluate_task(
                task=task,
                ordinal=ordinal,
                mode=mode,
                policy_client=policy_client,
                judge_client=judge_client,
                args=args,
            )
            append_jsonl(results_path, row)
            rows.append(row)
            atomic_json(output_dir / "summary.json", summarize_results(rows))
            print(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "status": row["status"],
                        "judge": (row.get("judge") or {}).get("verdict"),
                        "wall_seconds": row.get("wall_seconds"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await mode.teardown()
        await policy_client.close()
        if judge_client is not None:
            await judge_client.close()
    print(str(output_dir / "summary.json"))
    return 0


async def run_probe(args: argparse.Namespace) -> int:
    if args.env_file:
        parse_env_file(args.env_file.resolve())
    from lexbrowser_webvoyager_no_anti_bot.environment import LexmountDOMMode

    tasks = load_jsonl_tasks(args.tasks.resolve())
    if args.task_id:
        requested_ids = set(args.task_id)
        tasks = [task for task in tasks if task.task_id in requested_ids]
        missing_ids = sorted(requested_ids - {task.task_id for task in tasks})
        if missing_ids:
            raise RuntimeError(f"requested task ids are absent from manifest: {missing_ids}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("selected task manifest is empty")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            rows.append(row)
            task = row.get("task") or {}
            if row.get("backend") == args.backend and task.get("task_id"):
                completed_ids.add(str(task["task_id"]))

    if args.backend == "lexmount":
        api_key = _required_env("LEXMOUNT_API_KEY")
        project_id = _required_env("LEXMOUNT_PROJECT_ID")
    else:
        api_key = None
        project_id = None
    mode = LexmountDOMMode(
        api_key=api_key,
        project_id=project_id,
        browser_backend=args.backend,
        dom_backend="cdp",
        stagehand_model="",
        policy_model="",
        proxy_model_to_stagehand=False,
        browser_mode="normal",
        official_proxy=args.lexmount_official_proxy,
        external_proxy=None,
        local_chrome_executable_path=args.local_chrome_executable,
        local_chrome_headless=not args.local_chrome_headed,
        local_proxy_server=args.local_proxy_server,
        local_proxy_bypass=args.local_proxy_bypass,
        max_concurrent_sessions=args.concurrency,
        session_create_timeout_s=args.session_create_timeout,
        stagehand_ready_timeout_s=30.0,
        setup_navigation_timeout_s=args.setup_navigation_timeout,
        per_tool_timeout_s=args.per_tool_timeout,
        episode_timeout_s=args.episode_timeout,
        max_repeated_tool_calls=3,
    )
    manifest = {
        "schema_version": 1,
        "protocol": "webvoyager-browser-probe-v1",
        "started_at": utc_now(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "backend": args.backend,
        "tasks": str(args.tasks.resolve()),
        "tasks_sha256": sha256_file(args.tasks.resolve()),
        "selected_tasks": len(tasks),
        "evaluator": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "repository_revision": repository_revision(),
        },
        "browser": {
            "protocol": "fresh_session -> start_url -> observe",
            "dom_backend": "cdp",
            "setup_attempts": args.setup_attempts,
            "concurrency": args.concurrency,
            "local_chrome_headless": not args.local_chrome_headed,
            "local_proxy_configured": bool(args.local_proxy_server),
            "lexmount_official_proxy": args.lexmount_official_proxy,
        },
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    try:
        pending = [task for task in tasks if task.task_id not in completed_ids]
        probe_futures = [
            asyncio.create_task(probe_task(task=task, mode=mode, args=args))
            for task in pending
        ]
        for future in asyncio.as_completed(probe_futures):
            row = await future
            append_jsonl(results_path, row)
            rows.append(row)
            atomic_json(output_dir / "summary.json", summarize_probe_results(rows))
            print(
                json.dumps(
                    {
                        "task_id": (row.get("task") or {}).get("task_id"),
                        "status": row["status"],
                        "wall_seconds": row.get("wall_seconds"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await mode.teardown()
    print(str(output_dir / "summary.json"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-splits", help="create deterministic paired task manifests")
    prepare.add_argument("--training-parquet", type=Path, required=True)
    prepare.add_argument("--benchmark-jsonl", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--smoke-count", type=int, default=20)
    prepare.add_argument("--train-count", type=int, default=100)
    prepare.add_argument("--selection-seed", type=int, default=20260717)

    run = subparsers.add_parser("run", help="evaluate one checkpoint on one browser backend")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--backend", choices=("lexmount", "local"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--model-artifact", type=Path)
    run.add_argument("--model-sha256")
    run.add_argument("--env-file", type=Path)
    run.add_argument("--policy-base-url")
    run.add_argument("--policy-api-key")
    run.add_argument("--judge", choices=("off", "training"), default="off")
    run.add_argument("--judge-model", default="")
    run.add_argument("--judge-base-url")
    run.add_argument("--judge-api-key")
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--max-assistant-turns", type=int, default=6)
    run.add_argument("--seed", type=int, default=20260717)
    run.add_argument("--task-id", action="append", default=[])
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--setup-attempts", type=int, default=4)
    run.add_argument("--session-create-timeout", type=float, default=60.0)
    run.add_argument("--setup-navigation-timeout", type=float, default=30.0)
    run.add_argument("--per-tool-timeout", type=float, default=25.0)
    run.add_argument("--episode-timeout", type=float, default=180.0)
    run.add_argument("--local-chrome-executable")
    run.add_argument("--local-chrome-headed", action="store_true")
    run.add_argument("--local-proxy-server")
    run.add_argument("--local-proxy-bypass")
    run.add_argument("--lexmount-official-proxy", action="store_true")

    probe = subparsers.add_parser(
        "probe", help="measure browser usable-DOM availability without a policy"
    )
    probe.add_argument("--tasks", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--backend", choices=("lexmount", "local"), required=True)
    probe.add_argument("--env-file", type=Path)
    probe.add_argument("--task-id", action="append", default=[])
    probe.add_argument("--limit", type=int)
    probe.add_argument("--resume", action="store_true")
    probe.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum simultaneous browser sessions (default: 1)",
    )
    probe.add_argument("--setup-attempts", type=int, default=4)
    probe.add_argument("--session-create-timeout", type=float, default=60.0)
    probe.add_argument("--setup-navigation-timeout", type=float, default=30.0)
    probe.add_argument("--per-tool-timeout", type=float, default=25.0)
    probe.add_argument("--episode-timeout", type=float, default=180.0)
    probe.add_argument("--local-chrome-executable")
    probe.add_argument("--local-chrome-headed", action="store_true")
    probe.add_argument("--local-proxy-server")
    probe.add_argument("--local-proxy-bypass")
    probe.add_argument("--lexmount-official-proxy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-splits":
        return prepare_splits(args)
    if args.command == "run":
        return asyncio.run(run_evaluation(args))
    if args.command == "probe":
        return asyncio.run(run_probe(args))
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
