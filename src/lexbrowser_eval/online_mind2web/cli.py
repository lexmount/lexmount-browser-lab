#!/usr/bin/env python3
"""Reproducible Online-Mind2Web v2 evaluation on two browser backends.

The runner deliberately fails closed.  It only consumes the pinned official
Hugging Face dataset blob, emits strict ``online-mind2web-v2`` trajectories,
and invokes the pinned OSU WebJudge implementation.  Rollout concurrency is
fixed at ten for both browser backends; Judge concurrency is also ten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

BUBENCH_REPO = "https://github.com/lexmount/browseruse-agent-bench.git"
BUBENCH_COMMIT = "ccd5fcbdfb975257b2ce38161dc9bc2ab294b420"
OSU_REPO = "https://github.com/OSU-NLP-Group/Online-Mind2Web.git"
OSU_COMMIT = "f0d805ee0e9e0b3ea70911e45e5264b72968f3dc"
HF_REPO = "osunlp/Online-Mind2Web"
HF_REVISION = "84038480c979f3744ffadac18883b7095f90b332"
HF_FILENAME = "Online_Mind2Web.json"
HF_FILE_SIZE = 88380
HF_GIT_BLOB_OID = "ab2e7b3d33b66c7ba2973d0a972918efaa3dd114"
HF_JSON_SERIALIZATION_REPAIRS = (
    (
        b'first car in the list "best cars"?',
        b'first car in the list \\"best cars\\"?',
        "escape quoted phrase in task 47186fac8e7c7277af01144644eb4e0b_070826",
    ),
    (
        b'screen sizes from 55" to 64".',
        b'screen sizes from 55\\" to 64\\".',
        "escape inch marks in task 59912927c1fddee6ded8a49986896bc2_070826",
    ),
)

BENCHMARK = "Online-Mind2Web"
SPLIT = "All"
AGENT = "browser-use"
MODEL_CONFIG = "qwen3-8B"
MODEL_ID = "qwen3_8B"
JUDGE_MODEL = "gpt-5.4"
TASK_COUNT = 300
ROLLOUT_CONCURRENCY = 10
JUDGE_CONCURRENCY = 10
JUDGE_TEMPERATURE = 1
JUDGE_MAX_TOKENS = 512
JUDGE_MAX_TRIES = 3
JUDGE_SCORE_THRESHOLD = 3
MAX_STEPS = 40
TASK_TIMEOUT_SECONDS = 600
SCHEMA_VERSION = "online-mind2web-v2"

BACKENDS = {
    "lexmount": {"config_key": "lexmount", "result_id": "lexmount"},
    "local": {"config_key": "local", "result_id": "local"},
}

REQUIRED_ENV = (
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    "QWEN_MODEL_ID",
    "JUDGE_API_KEY",
    "JUDGE_BASE_URL",
    "LEXMOUNT_API_KEY",
    "LEXMOUNT_PROJECT_ID",
)

CONFIG_TEMPLATE = """\
default:
  agent: browser-use
  data: Online-Mind2Web
  model: qwen3-8B
  browser: lexmount
models:
  qwen3-8B:
    model_type: OPENAI
    model_provider: openai
    model_id: $QWEN_MODEL_ID
    api_key: $QWEN_API_KEY
    base_url: $QWEN_BASE_URL
    frequency_penalty: null
    dont_force_structured_output: false
    add_schema_to_system_prompt: true
browsers:
  lexmount:
    browser_id: lexmount
    lexmount_browser_mode: normal
    lexmount_official_proxy: false
    lexmount_api_key: $LEXMOUNT_API_KEY
    lexmount_project_id: $LEXMOUNT_PROJECT_ID
  local:
    browser_id: local
    headless: false
    local_proxy_server: ''
agents:
  browser-use:
    use_judge: false
    use_vision: false
    max_steps: 40
    flash_mode: true
    timeout: 600
site_skills:
  enabled: false
  dir: browseruse_bench/agents/site_skills/domain-skills
  max_chars: 30000
  max_files: 10
eval:
  model: gpt-5.4
  api_key: $JUDGE_API_KEY
  base_url: $JUDGE_BASE_URL
  temperature: 1
  max_tries: 3
  api_max_images: 50
  detail: high
  max_tokens: 512
"""


def log(message: str) -> None:
    print(f"[online-mind2web-v2] {message}", flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def run(
    command: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log("exec: " + " ".join(_redact_command(command)))
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        check=check,
    )


def output(command: Sequence[str], *, cwd: pathlib.Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout.strip()


def _redact_command(command: Sequence[str]) -> list[str]:
    redacted = list(command)
    for i, value in enumerate(redacted[:-1]):
        if value in {"--api-key", "--api_key", "--token"}:
            redacted[i + 1] = "<redacted>"
    return redacted


def atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def validate_environment(environ: Mapping[str, str]) -> dict[str, str]:
    missing = [name for name in REQUIRED_ENV if not environ.get(name, "").strip()]
    if missing:
        raise RuntimeError("missing environment variables: " + ", ".join(missing))
    values = {name: environ[name].strip() for name in REQUIRED_ENV}
    if values["QWEN_MODEL_ID"] != MODEL_ID:
        raise RuntimeError(f"QWEN_MODEL_ID must be {MODEL_ID!r}")
    return values


def resolve_policy_metadata(args: argparse.Namespace) -> dict[str, str | None]:
    """Record the evaluated checkpoint separately from its served API alias."""
    label = args.policy_label.strip()
    if not label:
        raise ValueError("--policy-label must not be empty")

    artifact_dir: pathlib.Path | None = None
    if args.policy_artifact is not None:
        artifact_dir = args.policy_artifact.expanduser().resolve()
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"policy artifact directory does not exist: {artifact_dir}")

    safetensors_sha256 = args.policy_sha256.strip().lower()
    if safetensors_sha256 and not re.fullmatch(r"[0-9a-f]{64}", safetensors_sha256):
        raise ValueError("--policy-sha256 must be a 64-character hexadecimal SHA-256")
    if artifact_dir is not None and safetensors_sha256:
        sidecar = artifact_dir / "model.safetensors.sha256"
        if sidecar.is_file():
            fields = sidecar.read_text(encoding="utf-8").strip().split()
            if not fields:
                raise ValueError(f"empty SHA-256 sidecar: {sidecar}")
            recorded = fields[0].lower()
            if recorded != safetensors_sha256:
                raise ValueError(
                    "--policy-sha256 does not match "
                    f"{sidecar}: expected {recorded}, got {safetensors_sha256}"
                )

    return {
        "label": label,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "safetensors_sha256": safetensors_sha256 or None,
    }


def _request_bytes(url: str, headers: Mapping[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError(
                    "the pinned official Hugging Face dataset is gated; set HF_TOKEN after "
                    "accepting its access terms, or set ONLINE_MIND2WEB_DATA_FILE to an exact "
                    "copy of the pinned file"
                ) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch pinned resource after 3 attempts: {last_error}")


def load_pinned_dataset(environ: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    supplied = environ.get("ONLINE_MIND2WEB_DATA_FILE", "").strip()
    if supplied:
        source_path = pathlib.Path(supplied).expanduser().resolve()
        data = source_path.read_bytes()
        source = f"local exact copy: {source_path}"
    else:
        token = (
            environ.get("HF_TOKEN", "").strip() or environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        )
        if not token:
            raise RuntimeError(
                "the pinned official Hugging Face dataset is gated and no HF_TOKEN was "
                "provided; alternatively set ONLINE_MIND2WEB_DATA_FILE to an exact copy "
                "of the pinned file"
            )
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILENAME}"
        data = _request_bytes(url, headers)
        source = f"{HF_REPO}@{HF_REVISION}/{HF_FILENAME}"

    actual_oid = git_blob_oid(data)
    if len(data) != HF_FILE_SIZE or actual_oid != HF_GIT_BLOB_OID:
        raise RuntimeError(
            "dataset blob mismatch: expected "
            f"size={HF_FILE_SIZE}, git_oid={HF_GIT_BLOB_OID}; got "
            f"size={len(data)}, git_oid={actual_oid}"
        )
    normalized_data = data
    applied_repairs: list[str] = []
    for invalid, valid, description in HF_JSON_SERIALIZATION_REPAIRS:
        count = normalized_data.count(invalid)
        if count != 1:
            raise RuntimeError(
                f"pinned official JSON repair precondition failed ({description}): "
                f"expected one occurrence, found {count}"
            )
        normalized_data = normalized_data.replace(invalid, valid)
        applied_repairs.append(description)
    payload = json.loads(normalized_data)
    if not isinstance(payload, list) or len(payload) != TASK_COUNT:
        raise RuntimeError(f"expected {TASK_COUNT} dataset rows")
    required = {"task_id", "confirmed_task", "website", "reference_length"}
    ids: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not required.issubset(item):
            raise RuntimeError(f"invalid official dataset row {index}")
        task_id = item["task_id"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise RuntimeError(f"invalid task_id at row {index}")
        if not isinstance(item["confirmed_task"], str) or not item["confirmed_task"].strip():
            raise RuntimeError(f"invalid confirmed_task at row {index}")
        if not isinstance(item["reference_length"], int) or item["reference_length"] < 1:
            raise RuntimeError(f"invalid reference_length at row {index}")
        ids.append(task_id)
    if len(set(ids)) != TASK_COUNT:
        raise RuntimeError("official dataset contains duplicate task IDs")
    manifest = {
        "repo": HF_REPO,
        "revision": HF_REVISION,
        "filename": HF_FILENAME,
        "source": source,
        "size": len(data),
        "git_blob_oid": actual_oid,
        "sha256": hashlib.sha256(data).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized_data).hexdigest(),
        "serialization_repairs": applied_repairs,
        "task_count": len(payload),
    }
    return payload, manifest


def verify_api_model(base_url: str, api_key: str, model_id: str) -> None:
    data = _request_bytes(
        base_url.rstrip("/") + "/models",
        {"Authorization": f"Bearer {api_key}"},
    )
    payload = json.loads(data)
    served = {
        str(item.get("id") or "") for item in payload.get("data", []) if isinstance(item, dict)
    }
    if model_id not in served:
        raise RuntimeError(f"model {model_id!r} is not served by {base_url}")


def bootstrap_checkout(
    repo: str, commit: str, checkout: pathlib.Path, sparse_paths: Sequence[str] = ()
) -> None:
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(checkout)])
    if sparse_paths:
        run(["git", "sparse-checkout", "init", "--cone"], cwd=checkout)
        run(["git", "sparse-checkout", "set", *sparse_paths], cwd=checkout)
    run(["git", "fetch", "--depth", "1", "origin", commit], cwd=checkout)
    run(["git", "checkout", "--detach", commit], cwd=checkout)
    actual = output(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != commit:
        raise RuntimeError(f"checkout mismatch: expected {commit}, got {actual}")


_OLD_SCREENSHOT_LOOP = """\
                # Save screenshots
                for i, b64_data in enumerate(history.screenshots() or [], 1):
                    if self.save_screenshot(b64_data, i, trajectory_dir):
                        screenshot_count += 1
                    elif b64_data:
                        logger.error(f\"Failed to save screenshot {i} for task {task_id}\")
"""

_V2_SCREENSHOT_LOOP = """\
                # Save screenshots without collapsing missing states.  The upstream
                # history.screenshots() helper filters null frames, which can desync
                # screenshots from api_logs/step_NNN.json.  This recording-only
                # adapter preserves the original history index required by v2.
                for i, hist_item in enumerate(history.history, 1):
                    state = getattr(hist_item, \"state\", None)
                    get_screenshot = getattr(state, \"get_screenshot\", None) if state else None
                    b64_data = get_screenshot() if callable(get_screenshot) else None
                    if self.save_screenshot(b64_data, i, trajectory_dir):
                        screenshot_count += 1
                    elif b64_data:
                        logger.error(f\"Failed to save screenshot {i} for task {task_id}\")
"""

_V3_SCREENSHOT_LOOP = """\
                # Count/save any history screenshots not already persisted by the
                # per-step callback.  Existing files are the authoritative exact
                # pre-action frames and must never be overwritten post-run.
                for i, hist_item in enumerate(history.history, 1):
                    screenshot_path = trajectory_dir / f"screenshot-{i}.png"
                    if screenshot_path.is_file():
                        screenshot_count += 1
                        continue
                    state = getattr(hist_item, "state", None)
                    get_screenshot = getattr(state, "get_screenshot", None) if state else None
                    b64_data = get_screenshot() if callable(get_screenshot) else None
                    if self.save_screenshot(b64_data, i, trajectory_dir):
                        screenshot_count += 1
                    elif b64_data:
                        logger.error(f"Failed to save screenshot {i} for task {task_id}")
"""

_V3_STEP_CALLBACK = """\
        async def _record_v2_step_screenshot(
            browser_state_summary: Any, _model_output: Any, step_number: int,
        ) -> None:
            # Persist the exact pre-action state while the browser is still open.
            # If BrowserUse's state capture timed out, retry through the same CDP
            # session; a neighbouring historical frame is never substituted.
            screenshot_path = trajectory_dir / f"screenshot-{step_number}.png"
            if screenshot_path.is_file():
                return
            get_screenshot = getattr(browser_state_summary, "get_screenshot", None)
            b64_data = get_screenshot() if callable(get_screenshot) else None
            if b64_data and self.save_screenshot(b64_data, step_number, trajectory_dir):
                return
            import base64
            for attempt in range(1, 4):
                try:
                    png_bytes = await asyncio.wait_for(browser.take_screenshot(), timeout=20)
                    b64_data = base64.b64encode(png_bytes).decode("ascii")
                    if self.save_screenshot(b64_data, step_number, trajectory_dir):
                        logger.info(
                            "Recovered exact screenshot %s for task %s on attempt %s",
                            step_number, task_id, attempt,
                        )
                        return
                except (OSError, RuntimeError, TimeoutError) as exc:
                    logger.warning(
                        "Exact screenshot retry %s/3 failed for task %s step %s: %s",
                        attempt, task_id, step_number, exc,
                    )
                if attempt < 3:
                    await asyncio.sleep(1)
            logger.error("No exact screenshot for task %s step %s", task_id, step_number)

"""

_V3_CALLBACK_MARKER = """\
        start_time = time.time()
        error_msg = None

        try:
"""

_V3_AGENT_ARGUMENT_MARKER = """\
                use_judge=_get_config_value(agent_config, "use_judge", "USE_JUDGE", default=False),
            )
"""

_V3_AGENT_ARGUMENT = """\
                use_judge=_get_config_value(agent_config, "use_judge", "USE_JUDGE", default=False),
                register_new_step_callback=_record_v2_step_screenshot,
            )
"""

_BROKEN_V2_SCREENSHOT_LOOP = _V2_SCREENSHOT_LOOP.replace(
    '                    get_screenshot = getattr(state, "get_screenshot", None) if state else None\n'
    "                    b64_data = get_screenshot() if callable(get_screenshot) else None\n",
    '                    b64_data = getattr(state, "screenshot", None) if state else None\n',
)


def apply_v2_recording_adapter(checkout: pathlib.Path) -> dict[str, str]:
    path = checkout / "browseruse_bench/agents/browser_use.py"
    original = path.read_text(encoding="utf-8")
    clean = output(
        ["git", "show", f"{BUBENCH_COMMIT}:browseruse_bench/agents/browser_use.py"], cwd=checkout
    )
    # git show strips the final newline through output(); restore it for exact replacement.
    if not clean.endswith("\n"):
        clean += "\n"
    known_adapter = any(
        block in original
        for block in (
            _BROKEN_V2_SCREENSHOT_LOOP,
            _V2_SCREENSHOT_LOOP,
            _V3_SCREENSHOT_LOOP,
        )
    )
    if original != clean and not known_adapter:
        raise RuntimeError(f"unexpected pre-existing modification: {path}")
    source = original.replace(_BROKEN_V2_SCREENSHOT_LOOP, _V2_SCREENSHOT_LOOP)
    if _V3_SCREENSHOT_LOOP not in source:
        if _V2_SCREENSHOT_LOOP in source:
            source = source.replace(_V2_SCREENSHOT_LOOP, _V3_SCREENSHOT_LOOP)
        elif _OLD_SCREENSHOT_LOOP in source:
            source = source.replace(_OLD_SCREENSHOT_LOOP, _V3_SCREENSHOT_LOOP)
        else:
            raise RuntimeError("cannot locate the pinned screenshot recording block")
    if _V3_STEP_CALLBACK not in source:
        if source.count(_V3_CALLBACK_MARKER) != 1:
            raise RuntimeError("cannot locate BrowserUse per-step callback insertion point")
        source = source.replace(
            _V3_CALLBACK_MARKER,
            _V3_STEP_CALLBACK + _V3_CALLBACK_MARKER,
        )
    if "register_new_step_callback=_record_v2_step_screenshot" not in source:
        if source.count(_V3_AGENT_ARGUMENT_MARKER) != 1:
            raise RuntimeError("cannot locate BrowserUse Agent callback argument")
        source = source.replace(_V3_AGENT_ARGUMENT_MARKER, _V3_AGENT_ARGUMENT)
    path.write_text(source, encoding="utf-8")
    patched = path.read_bytes()
    return {
        "kind": "recording-only v2 exact per-step screenshot adapter",
        "path": "browseruse_bench/agents/browser_use.py",
        "sha256": hashlib.sha256(patched).hexdigest(),
        "capture_policy": (
            "persist the exact pre-action BrowserState in register_new_step_callback; "
            "if missing, retry the same live CDP session up to three times"
        ),
    }


def write_dataset_for_bubench(
    checkout: pathlib.Path, tasks: Iterable[dict[str, Any]]
) -> pathlib.Path:
    path = checkout / "browseruse_bench/data/Online-Mind2Web/task.jsonl"
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in tasks)
    path.write_text(text, encoding="utf-8")
    return path


def write_config(path: pathlib.Path, checkout: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (path, checkout / "config.yaml"):
        target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        target.chmod(0o600)


def install_bubench(checkout: pathlib.Path, uv: str) -> None:
    bubench = checkout / ".venv/bin/bubench"
    if not bubench.exists():
        run([uv, "sync"], cwd=checkout)


def install_osu_judge(checkout: pathlib.Path, uv: str) -> pathlib.Path:
    python = checkout / ".venv/bin/python"
    marker = checkout / ".venv/.online-mind2web-requirements-ready"
    requirements = checkout / "requirements.txt"
    requirements_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not python.exists():
        run([uv, "venv", "--python", "3.11", str(checkout / ".venv")], cwd=checkout)
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != requirements_hash:
        run([uv, "pip", "install", "--python", str(python), "-r", str(requirements)], cwd=checkout)
        marker.write_text(requirements_hash + "\n", encoding="utf-8")
    return python


def osu_integrity(checkout: pathlib.Path) -> dict[str, str]:
    files = (
        "src/run.py",
        "src/utils.py",
        "src/methods/webjudge_online_mind2web.py",
        "script/eval.sh",
    )
    result = {}
    for relative in files:
        expected = output(["git", "rev-parse", f"{OSU_COMMIT}:{relative}"], cwd=checkout)
        actual = output(["git", "hash-object", relative], cwd=checkout)
        if actual != expected:
            raise RuntimeError(f"official OSU Judge file was modified: {relative}")
        result[relative] = actual
    return result


def prepare_judge_source(
    osu_checkout: pathlib.Path, campaign_dir: pathlib.Path, source_tag: str = "default"
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Create an isolated official-Judge source tree with the declared temperature deviation.

    The pinned OSU checkout remains byte-for-byte intact and continues to pass
    ``osu_integrity``.  Only the two default values that control
    ``OpenaiEngine.generate`` are changed in the campaign-local copy; prompts,
    max tokens, retry policy, and evaluation logic are untouched.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_tag):
        raise ValueError(f"invalid Judge source tag: {source_tag}")
    target = campaign_dir / (f"judge_source_temperature_{JUDGE_TEMPERATURE}_{source_tag}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(osu_checkout / "src", target / "src")
    utils_path = target / "src/utils.py"
    source = utils_path.read_text(encoding="utf-8")
    init_old = "        temperature=0,\n        port=-1,"
    generate_old = (
        "    def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs):"
    )
    if source.count(init_old) != 1 or source.count(generate_old) != 1:
        raise RuntimeError("pinned OSU temperature patch precondition failed")
    source = source.replace(
        init_old,
        f"        temperature={JUDGE_TEMPERATURE},\n        port=-1,",
    ).replace(
        generate_old,
        "    def generate(self, messages, max_new_tokens=512, "
        f"temperature={JUDGE_TEMPERATURE}, model=None, **kwargs):",
    )
    utils_path.write_text(source, encoding="utf-8")
    patch_manifest = {
        "kind": "declared Judge temperature-only deviation",
        "official_source": str(osu_checkout),
        "isolated_source": str(target),
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "max_tries": JUDGE_MAX_TRIES,
        "changed_file": "src/utils.py",
        "sha256": hashlib.sha256(utils_path.read_bytes()).hexdigest(),
    }
    atomic_json(target / "patch_manifest.json", patch_manifest)
    return target, patch_manifest


def build_rollout_command(
    checkout: pathlib.Path,
    config: pathlib.Path,
    backend: str,
    timestamp: str,
    rollout_concurrency: int = ROLLOUT_CONCURRENCY,
) -> list[str]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    command = [
        str(checkout / ".venv/bin/bubench"),
        "run",
        "--agent",
        AGENT,
        "--data",
        BENCHMARK,
        "--split",
        SPLIT,
        "--agent-config",
        str(config),
        "--model",
        MODEL_CONFIG,
        "--browser",
        BACKENDS[backend]["config_key"],
        "--mode",
        "all",
        "--concurrency",
        str(rollout_concurrency),
        "--skip-completed",
        "--timestamp",
        timestamp,
    ]
    if backend == "local":
        command = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", *command]
    return command


def official_run_dir(checkout: pathlib.Path, timestamp: str) -> pathlib.Path:
    return checkout / "experiments" / BENCHMARK / SPLIT / AGENT / MODEL_ID / timestamp


def _json_text(value: Any, limit: int = 600) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _status(result: dict[str, Any] | None) -> str:
    return "FAILED" if result and result.get("error") else "SUCCESS"


def _param(action: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(action, dict) or not action:
        return "unknown", {"raw": action}
    name, value = next(iter(action.items()))
    if isinstance(value, dict):
        return str(name), value
    return str(name), {"value": value}


def format_action(
    action: Any, action_result: dict[str, Any] | None, current_url: str | None
) -> tuple[str, str | None, str | None, str | None]:
    """Return action text, status, URL override, and terminal answer."""
    name, params = _param(action)
    normalized = name.lower()
    status = _status(action_result)
    suffix = f" | {status}"
    index = params.get("index")
    selector = f"[browser-use-index='{index}']" if index is not None else "page"
    terminal_answer: str | None = None
    url_override: str | None = None

    if normalized == "done":
        terminal_answer = str(params.get("text") or params.get("value") or "").strip()
        return f"TASK_COMPLETE -> ANSWER: {terminal_answer}", None, current_url, terminal_answer
    if normalized in {"navigate", "open_tab"}:
        url_override = str(params.get("url") or current_url or "")
        return f"page -> NAVIGATE -> navigate to {url_override}{suffix}", status, url_override, None
    if normalized == "search_google":
        query = params.get("query") or params.get("value") or ""
        return (
            f"page -> NAVIGATE -> search Google for {_json_text(query)}{suffix}",
            status,
            current_url,
            None,
        )
    if normalized in {"click", "click_element"}:
        return (
            f"CLICK {selector} -> click browser element {_json_text(params)}{suffix}",
            status,
            current_url,
            None,
        )
    if normalized in {"input_text", "type", "type_text"}:
        text = params.get("text") or params.get("value") or ""
        return f"TYPE {selector} -> type {_json_text(text)}{suffix}", status, current_url, None
    if normalized in {"scroll", "scroll_down", "scroll_up"}:
        return f"SCROLL page -> scroll {_json_text(params)}{suffix}", status, current_url, None
    if normalized in {"send_keys", "press_key"}:
        keys = params.get("keys") or params.get("key") or params.get("value") or ""
        return f"PRESS_KEY page -> press {_json_text(keys)}{suffix}", status, current_url, None
    if normalized in {"go_back", "back"}:
        return f"page -> GO_BACK -> return to the previous page{suffix}", status, current_url, None
    if normalized in {"go_forward", "forward"}:
        return f"page -> GO_FORWARD -> move to the next page{suffix}", status, current_url, None
    if normalized in {"reload", "refresh"}:
        return f"page -> REFRESH -> reload the current page{suffix}", status, current_url, None
    if normalized in {"wait"}:
        return f"WAIT page -> wait {_json_text(params)}{suffix}", status, current_url, None
    if normalized in {"select_dropdown_option", "select_option", "select"}:
        return (
            f"SELECT {selector} -> select {_json_text(params)}{suffix}",
            status,
            current_url,
            None,
        )
    if normalized in {"hover"}:
        return (
            f"HOVER {selector} -> hover over browser element {_json_text(params)}{suffix}",
            status,
            current_url,
            None,
        )
    if normalized in {"switch_tab", "close_tab"}:
        return (
            f"CLICK page -> {normalized.replace('_', ' ')} {_json_text(params)}{suffix}",
            status,
            current_url,
            None,
        )
    if normalized in {"extract_content", "get_dropdown_options"}:
        return (
            f"WAIT page -> observe page content for {_json_text(params)}{suffix}",
            status,
            current_url,
            None,
        )
    return (
        f"WAIT page -> perform browser action {normalized} {_json_text(params)}{suffix}",
        status,
        current_url,
        None,
    )


def _thought(step_log: dict[str, Any]) -> str | None:
    model_output = step_log.get("output") or {}
    parts = []
    for key in ("thinking", "next_goal"):
        value = model_output.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return "\n".join(parts) if parts else None


def validate_v2(
    payload: Any, task_dir: pathlib.Path, official_task: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    allowed_top = {
        "schema_version",
        "task",
        "task_id",
        "agent_final_answer",
        "reference_length",
        "action_history",
    }
    if not isinstance(payload, dict):
        return ["result is not an object"]
    if set(payload) - allowed_top:
        errors.append("unexpected top-level properties")
    for key in ("schema_version", "task", "task_id", "reference_length", "action_history"):
        if key not in payload:
            errors.append(f"missing {key}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        errors.append("invalid task_id")
    if official_task:
        if payload.get("task_id") != official_task.get("task_id"):
            errors.append("official task_id mismatch")
        if payload.get("task") != official_task.get("confirmed_task"):
            errors.append("task is not the verbatim official user task")
        if payload.get("reference_length") != official_task.get("reference_length"):
            errors.append("reference_length mismatch")
    actions = payload.get("action_history")
    if not isinstance(actions, list) or not actions:
        errors.append("action_history is empty")
        return errors
    allowed_step = {"step", "screenshot", "url", "action", "thought", "action_status"}
    for index, step in enumerate(actions):
        if not isinstance(step, dict):
            errors.append(f"step {index} is not an object")
            continue
        if set(step) - allowed_step or not {"step", "screenshot", "action", "thought"}.issubset(
            step
        ):
            errors.append(f"step {index} shape mismatch")
        if step.get("step") != index:
            errors.append(f"step index mismatch at {index}")
        screenshot = step.get("screenshot")
        if not isinstance(screenshot, str) or pathlib.Path(screenshot).name != screenshot:
            errors.append(f"invalid screenshot at step {index}")
        elif not (task_dir / "trajectory" / screenshot).is_file():
            errors.append(f"missing screenshot at step {index}")
        if "thought" not in step or (
            step.get("thought") is not None and not isinstance(step.get("thought"), str)
        ):
            errors.append(f"invalid thought at step {index}")
        if step.get("action_status") not in {None, "SUCCESS", "FAILED"}:
            errors.append(f"invalid action_status at step {index}")
    last_step = actions[-1]
    if not isinstance(last_step, dict):
        errors.append("final step is not a v2 object")
        return errors
    last_action = str(last_step.get("action") or "")
    if not last_action.startswith("TASK_COMPLETE -> ANSWER:"):
        errors.append("missing terminal TASK_COMPLETE")
    else:
        answer = last_action.split("TASK_COMPLETE -> ANSWER:", 1)[1].strip()
        if payload.get("agent_final_answer") != answer:
            errors.append("agent_final_answer mismatch")
    return errors


def convert_task_result(task_dir: pathlib.Path, official_task: dict[str, Any]) -> tuple[bool, str]:
    result_path = task_dir / "result.json"
    if not result_path.is_file() or not result_path.stat().st_size:
        return False, "missing result"
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result_path.unlink(missing_ok=True)
        return False, "malformed result"
    existing_errors = validate_v2(raw, task_dir, official_task)
    if not existing_errors:
        return True, "already valid"

    if not isinstance(raw, dict) or raw.get("schema_version") == SCHEMA_VERSION:
        result_path.unlink(missing_ok=True)
        return False, "invalid v2 result"
    answer_text = str(raw.get("answer") or "")
    if "process interrupted" in answer_text.lower():
        result_path.replace(task_dir / "interrupted_result.json")
        return False, "interrupted placeholder excluded"

    logs = []
    for path in sorted((task_dir / "api_logs").glob("step_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            logs.append(item)
    if not logs:
        result_path.replace(task_dir / "bubench_result.json")
        return False, "no structured step logs"

    trajectory = task_dir / "trajectory"
    normalized_dir = task_dir / ".trajectory-v2"
    if normalized_dir.exists():
        shutil.rmtree(normalized_dir)
    normalized_dir.mkdir()
    action_history: list[dict[str, Any]] = []
    final_answer: str | None = None
    last_url: str | None = None
    last_source: pathlib.Path | None = None

    for log_index, step_log in enumerate(logs, 1):
        input_data = step_log.get("input") or {}
        current_url = input_data.get("url") if isinstance(input_data.get("url"), str) else last_url
        if current_url:
            last_url = current_url
        actions = (step_log.get("output") or {}).get("actions") or []
        source = trajectory / f"screenshot-{log_index}.png"
        if not source.is_file():
            # browser-use records its automatic initial NAVIGATE before the
            # first browser-state screenshot.  The next frame is the factual
            # post-navigation state and is the only correct evidence for that
            # special step; no other missing frame may be substituted.
            first_action_name = _param(actions[0])[0].lower() if actions else ""
            post_navigation = trajectory / "screenshot-2.png"
            if log_index == 1 and first_action_name == "navigate" and post_navigation.is_file():
                source = post_navigation
            else:
                result_path.replace(task_dir / "bubench_result.json")
                shutil.rmtree(normalized_dir)
                return False, f"missing exact state screenshot for api step {log_index}"
        last_source = source
        results = step_log.get("action_results") or []
        if not actions:
            actions = [{"wait": {"reason": "model observation without a browser action"}}]
        for action_index, action in enumerate(actions):
            result = (
                results[action_index]
                if action_index < len(results) and isinstance(results[action_index], dict)
                else None
            )
            text, status, url, terminal = format_action(action, result, current_url)
            screenshot_name = f"{len(action_history):04d}.png"
            shutil.copyfile(source, normalized_dir / screenshot_name)
            step = {
                "step": len(action_history),
                "screenshot": screenshot_name,
                "url": url,
                "action": text,
                "thought": _thought(step_log),
                "action_status": status,
            }
            action_history.append(step)
            if terminal is not None:
                final_answer = terminal

    if not action_history or last_source is None:
        result_path.replace(task_dir / "bubench_result.json")
        shutil.rmtree(normalized_dir)
        return False, "empty structured trajectory"

    if not action_history[-1]["action"].startswith("TASK_COMPLETE -> ANSWER:"):
        # Do not fabricate a terminal action for timeout/max-steps/runtime
        # failures.  Only an Agent-produced `done` action is a valid v2
        # terminal and therefore eligible for --skip-completed or Judge.
        result_path.replace(task_dir / "unterminated_result.json")
        shutil.rmtree(normalized_dir)
        return False, "agent produced no valid TASK_COMPLETE terminal"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": official_task["confirmed_task"],
        "task_id": official_task["task_id"],
        "agent_final_answer": final_answer,
        "reference_length": official_task["reference_length"],
        "action_history": action_history,
    }
    # Swap screenshot folders only after the complete v2 object is assembled.
    raw_trajectory = task_dir / "trajectory-bubench"
    if raw_trajectory.exists():
        shutil.rmtree(raw_trajectory)
    trajectory.replace(raw_trajectory)
    normalized_dir.replace(trajectory)
    errors = validate_v2(payload, task_dir, official_task)
    if errors:
        trajectory.replace(normalized_dir)
        raw_trajectory.replace(trajectory)
        shutil.rmtree(normalized_dir, ignore_errors=True)
        result_path.replace(task_dir / "bubench_result.json")
        return False, "; ".join(errors)
    result_path.replace(task_dir / "bubench_result.json")
    atomic_json(result_path, payload)
    return True, "converted"


def normalize_run(run_dir: pathlib.Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["task_id"]: item for item in tasks}
    valid: list[str] = []
    invalid: dict[str, str] = {}
    for task_id, official_task in by_id.items():
        task_dir = run_dir / "tasks" / task_id
        if not task_dir.is_dir():
            invalid[task_id] = "missing task directory"
            continue
        ok, reason = convert_task_result(task_dir, official_task)
        if ok:
            valid.append(task_id)
        else:
            invalid[task_id] = reason
    return {
        "checked_at": utc_now(),
        "run_dir": str(run_dir),
        "valid_count": len(valid),
        "valid_task_ids": sorted(valid),
        "invalid_count": len(invalid),
        "invalid": invalid,
        "complete": len(valid) == len(tasks) and not invalid,
    }


def judge_output_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / (
        f"WebJudge_Online_Mind2Web_eval_{JUDGE_MODEL}_score_threshold_"
        f"{JUDGE_SCORE_THRESHOLD}_auto_eval_results.json"
    )


def forced_failures_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / "forced_failures.json"


def load_forced_failures(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(task_id): details
        for task_id, details in payload.items()
        if task_id and isinstance(details, dict)
    }


def load_judge_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def inspect_judge(path: pathlib.Path, expected_ids: set[str]) -> dict[str, Any]:
    records = load_judge_records(path)
    ids = [str(item.get("task_id") or "") for item in records]
    counts = Counter(ids)
    unique = set(ids) - {""}
    duplicates = sorted(task_id for task_id, count in counts.items() if task_id and count > 1)
    forced = load_forced_failures(forced_failures_path(path.parent))
    forced_ids = set(forced)
    labels = [
        item.get("predicted_label") for item in records if item.get("task_id") in expected_ids
    ]
    invalid_labels = sum(label not in {0, 1} for label in labels)
    success = sum(label == 1 for label in labels)
    overlap = sorted(unique & forced_ids)
    covered = unique | forced_ids
    return {
        "checked_at": utc_now(),
        "path": str(path),
        "record_count": len(records),
        "official_unique_count": len(unique),
        "forced_failure_count": len(forced_ids),
        "unique_count": len(covered),
        "missing_task_ids": sorted(expected_ids - covered),
        "unexpected_task_ids": sorted(covered - expected_ids),
        "duplicate_task_ids": duplicates,
        "forced_failure_task_ids": sorted(forced_ids),
        "official_forced_overlap": overlap,
        "invalid_label_count": invalid_labels,
        "successful_tasks": success,
        "failed_tasks": len(labels) - success + len(forced_ids),
        "success_rate": (100.0 * success / len(expected_ids)) if expected_ids else 0.0,
        "complete": (
            covered == expected_ids and not duplicates and not invalid_labels and not overlap
        ),
    }


CONTENT_POLICY_MARKERS = (
    "content_policy_violation",
    "contentpolicyviolationerror",
    "content safety system",
    "input image may contain content that is not allowed",
)


def _run_isolated_judge_task(
    osu_checkout: pathlib.Path,
    python: pathlib.Path,
    tasks_dir: pathlib.Path,
    retry_root: pathlib.Path,
    task_id: str,
    api_key: str,
    judge_env: Mapping[str, str],
) -> tuple[str, dict[str, Any] | None, str | None, str]:
    """Retry one missing task with its own official evaluator process.

    Up to ten of these calls run concurrently.  Per-task isolation makes a
    content-policy rejection attributable without changing OSU prompts or
    evaluator code.
    """
    task_root = retry_root / task_id
    trajectories = task_root / "input"
    output_dir = task_root / "output"
    shutil.rmtree(task_root, ignore_errors=True)
    trajectories.mkdir(parents=True)
    output_dir.mkdir()
    (trajectories / task_id).symlink_to(tasks_dir / task_id, target_is_directory=True)
    command = build_judge_command(
        osu_checkout,
        python,
        trajectories,
        output_dir,
        api_key,
    )
    command[command.index("--num_worker") + 1] = "1"
    completed = subprocess.run(
        command,
        cwd=str(osu_checkout),
        env=dict(judge_env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_text = completed.stdout or ""
    records = load_judge_records(judge_output_path(output_dir))
    record = next(
        (item for item in records if item.get("task_id") == task_id),
        None,
    )
    lowered = output_text.lower()
    reason = None
    if record is None and any(marker in lowered for marker in CONTENT_POLICY_MARKERS):
        reason = "content_policy_rejection_after_official_retries"
    return task_id, record, reason, output_text


def retry_missing_judge_tasks_isolated(
    osu_checkout: pathlib.Path,
    python: pathlib.Path,
    tasks_dir: pathlib.Path,
    output_dir: pathlib.Path,
    missing: Sequence[str],
    api_key: str,
    judge_env: Mapping[str, str],
) -> None:
    if not missing:
        return
    retry_root = output_dir / ".isolated-retries"
    retry_root.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, dict[str, Any] | None, str | None, str]] = []
    with ThreadPoolExecutor(max_workers=min(JUDGE_CONCURRENCY, len(missing))) as pool:
        futures = [
            pool.submit(
                _run_isolated_judge_task,
                osu_checkout,
                python,
                tasks_dir,
                retry_root,
                task_id,
                api_key,
                judge_env,
            )
            for task_id in missing
        ]
        for future in as_completed(futures):
            results.append(future.result())

    judge_file = judge_output_path(output_dir)
    forced_path = forced_failures_path(output_dir)
    forced = load_forced_failures(forced_path)
    with judge_file.open("a", encoding="utf-8") as stream:
        for task_id, record, reason, output_text in sorted(results):
            if output_text:
                print(output_text, end="" if output_text.endswith("\n") else "\n")
            if record is not None and record.get("predicted_label") in {0, 1}:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                forced.pop(task_id, None)
            elif reason:
                forced[task_id] = {
                    "predicted_label": 0,
                    "reason": reason,
                    "recorded_at": utc_now(),
                    "policy": "user-authorized infrastructure failure",
                }
    atomic_json(forced_path, forced)
    shutil.rmtree(retry_root, ignore_errors=True)


def build_judge_command(
    osu_checkout: pathlib.Path,
    python: pathlib.Path,
    trajectories: pathlib.Path,
    output_dir: pathlib.Path,
    api_key: str,
) -> list[str]:
    return [
        str(python),
        str(osu_checkout / "src/run.py"),
        "--mode",
        "WebJudge_Online_Mind2Web_eval",
        "--model",
        JUDGE_MODEL,
        "--trajectories_dir",
        str(trajectories),
        "--api_key",
        api_key,
        "--output_path",
        str(output_dir),
        "--score_threshold",
        str(JUDGE_SCORE_THRESHOLD),
        "--num_worker",
        str(JUDGE_CONCURRENCY),
    ]


def run_official_judge_isolated(
    osu_checkout: pathlib.Path,
    python: pathlib.Path,
    tasks_dir: pathlib.Path,
    output_dir: pathlib.Path,
    expected_ids: set[str],
    api_key: str,
    judge_env: Mapping[str, str],
) -> dict[str, Any]:
    """Run the official evaluator in bounded batches of up to ten tasks.

    Batching avoids OSU's ``chunk_size == 0`` edge case on a final short batch
    and caps the actual number of worker processes at the declared Judge
    concurrency without changing prompts, retries, or output format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_file = judge_output_path(output_dir)
    audit = inspect_judge(judge_file, expected_ids)
    missing = list(audit["missing_task_ids"])
    staging_root = output_dir / ".single-task-input"
    batches = [
        missing[i : i + JUDGE_CONCURRENCY] for i in range(0, len(missing), JUDGE_CONCURRENCY)
    ]
    for index, batch in enumerate(batches, 1):
        if staging_root.exists() or staging_root.is_symlink():
            shutil.rmtree(staging_root)
        staging_root.mkdir()
        for task_id in batch:
            source = tasks_dir / task_id
            if not source.is_dir():
                raise RuntimeError(f"missing rollout directory for Judge task {task_id}")
            (staging_root / task_id).symlink_to(source, target_is_directory=True)
        log(f"Judge batch {index}/{len(batches)}: {len(batch)} tasks")
        command = build_judge_command(
            osu_checkout,
            python,
            staging_root,
            output_dir,
            api_key,
        )
        command[command.index("--num_worker") + 1] = str(len(batch))
        run(
            command,
            cwd=osu_checkout,
            env=judge_env,
            check=False,
        )
        audit = inspect_judge(judge_file, expected_ids)
    if audit["missing_task_ids"]:
        log(
            "Judge isolated retry for missing tasks with concurrency="
            f"{JUDGE_CONCURRENCY}: {len(audit['missing_task_ids'])} tasks"
        )
        retry_missing_judge_tasks_isolated(
            osu_checkout,
            python,
            tasks_dir,
            output_dir,
            audit["missing_task_ids"],
            api_key,
            judge_env,
        )
        audit = inspect_judge(judge_file, expected_ids)
    shutil.rmtree(staging_root, ignore_errors=True)
    return audit


def write_report(
    path: pathlib.Path,
    backend: str,
    campaign: str,
    dataset: dict[str, Any],
    rollout: dict[str, Any],
    judge: dict[str, Any],
    run_dir: pathlib.Path,
    policy_model: Mapping[str, str | None],
    rollout_concurrency: int = ROLLOUT_CONCURRENCY,
    task_count: int = TASK_COUNT,
) -> None:
    if not rollout.get("complete") or not judge.get("complete"):
        raise RuntimeError("refusing to publish an incomplete formal report")
    label = "Lexmount" if backend == "lexmount" else "Chrome-Local (5090)"
    text = f"""# Online-Mind2Web — {policy_model["label"]} + {label}

## 结果

| 项目 | 数值 |
|---|---:|
| Tasks | {task_count}/{task_count} |
| WebJudge 成功 | {judge["successful_tasks"]} |
| WebJudge 失败 | {judge["failed_tasks"]} |
| 内容策略强制失败 | {judge.get("forced_failure_count", 0)} |
| Success rate | {judge["success_rate"]:.2f}% |
| Rollout concurrency | {rollout_concurrency} |
| Judge concurrency | {JUDGE_CONCURRENCY} |

## 固定配置与可复现性

- Campaign: `{campaign}`
- Policy checkpoint: `{policy_model["label"]}`
- Served endpoint model ID: `{policy_model["endpoint_model_id"]}`
- Policy artifact: `{policy_model["artifact_dir"] or "not recorded"}`
- Policy safetensors SHA-256: `{policy_model["safetensors_sha256"] or "not recorded"}`
- Browser backend: `{label}`
- Bubench revision: `{BUBENCH_COMMIT}`
- OSU evaluator revision: `{OSU_COMMIT}`
- Hugging Face dataset revision: `{dataset["revision"]}`
- Dataset git blob: `{dataset["git_blob_oid"]}`
- Dataset SHA-256: `{dataset["sha256"]}`
- Dataset serialization: 原始官方 blob 原样锁定；仅修复其两处未转义双引号以恢复合法 JSON，任务语义和全部字段值不变（详见 campaign manifest）
- Schema: `{SCHEMA_VERSION}`，{task_count}/{task_count} 均通过逐步 action/thought/screenshot/URL 与终态校验
- Agent defaults: `use_vision=false`, `max_steps={MAX_STEPS}`, `flash_mode=true`, `timeout={TASK_TIMEOUT_SECONDS}`
- WebJudge: 官方 OSU prompt/流程，`temperature={JUDGE_TEMPERATURE}`, `max_tokens={JUDGE_MAX_TOKENS}`, `max_tries={JUDGE_MAX_TRIES}`, threshold={JUDGE_SCORE_THRESHOLD}

## 明确偏差

采用官方 WebJudge 流程，但 Judge backbone 非官方推荐的 o4-mini；本次按实验要求使用 `{JUDGE_MODEL}`。Judge concurrency 固定为 `{JUDGE_CONCURRENCY}`，temperature 按最新实验授权设为 `{JUDGE_TEMPERATURE}`。除此之外，Judge prompt、max_tokens、重试策略与 OSU 官方实现保持一致。

## 产物

- 远端正式轨迹：`{run_dir}`
- Judge JSONL：`{judge["path"]}`
- 基础设施强制失败：`{forced_failures_path(pathlib.Path(judge["path"]).parent)}`

> Online-Mind2Web 运行于实时公网，网页内容、地区跳转、验证码和站点可用性会随时间变化；结果应与上述 dataset/code revision 及运行时间绑定解读。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_comparison_report(
    path: pathlib.Path,
    campaign: str,
    states: Mapping[str, tuple[pathlib.Path, dict[str, Any], dict[str, Any] | None]],
    policy_model: Mapping[str, str | None],
    rollout_concurrency: int = ROLLOUT_CONCURRENCY,
    task_count: int = TASK_COUNT,
) -> None:
    if set(states) != set(BACKENDS):
        return
    rows: dict[str, dict[str, Any]] = {}
    for backend, (_, rollout, judge) in states.items():
        if not rollout.get("complete") or not judge or not judge.get("complete"):
            raise RuntimeError("refusing to publish an incomplete comparison report")
        rows[backend] = judge
    lexmount = rows["lexmount"]
    local = rows["local"]
    delta = local["success_rate"] - lexmount["success_rate"]
    text = f"""# Online-Mind2Web：Lexmount Browser 与 Local Chrome 对比

本轮使用同一 `{policy_model["label"]}`、固定 {task_count}-task 数据集切片、相同 Agent 配置和 rollout concurrency={rollout_concurrency}。两端只运行官方 `WebJudge_Online_Mind2Web_eval` 分支，Judge backbone 为 gpt-5.4、temperature=1、concurrency=10。内容策略拒绝且经官方重试仍无记录的任务按用户批准计失败。

## 最终质量指标

| 指标 | Lexmount Browser | Local Chrome | Local 相对 Lexmount |
|---|---:|---:|---:|
| Success | **{lexmount["success_rate"]:.2f}%（{lexmount["successful_tasks"]}/{task_count}）** | **{local["success_rate"]:.2f}%（{local["successful_tasks"]}/{task_count}）** | {delta:+.2f} pp |
| Failure | {lexmount["failed_tasks"]} | {local["failed_tasks"]} | {local["failed_tasks"] - lexmount["failed_tasks"]:+d} |
| 内容策略强制失败 | {lexmount.get("forced_failure_count", 0)} | {local.get("forced_failure_count", 0)} | {local.get("forced_failure_count", 0) - lexmount.get("forced_failure_count", 0):+d} |
| 严格有效轨迹 | {task_count}/{task_count} | {task_count}/{task_count} | — |

## 评测口径

- Campaign: `{campaign}`
- Policy checkpoint: `{policy_model["label"]}`
- Served endpoint model ID: `{policy_model["endpoint_model_id"]}`
- Policy artifact: `{policy_model["artifact_dir"] or "not recorded"}`
- Policy safetensors SHA-256: `{policy_model["safetensors_sha256"] or "not recorded"}`
- Dataset: `osunlp/Online-Mind2Web@{HF_REVISION}`，{task_count} tasks（固定前 {task_count} 条；全量300条先完成哈希与结构校验）
- Schema: `{SCHEMA_VERSION}`
- Rollout: qwen3-8B，concurrency={rollout_concurrency}，两后端同配置
- Judge: 仅 `WebJudge_Online_Mind2Web_eval`，gpt-5.4，temperature={JUDGE_TEMPERATURE}，concurrency={JUDGE_CONCURRENCY}
- 官方 Judge 参数：max_tokens={JUDGE_MAX_TOKENS}，max_tries={JUDGE_MAX_TRIES}，score_threshold={JUDGE_SCORE_THRESHOLD}
- 指标：`Success Rate = predicted_label=1 / {task_count} × 100%`

## 明确偏差

采用官方 OSU WebJudge prompt、流程和标签解析，但 Judge backbone 非官方推荐的 o4-mini；temperature 与并发按本轮实验授权分别设为 1 和 10。无法形成官方 Judge 记录的内容策略拒绝单独保存在 `forced_failures.json`，计为失败而不伪造官方 JSONL。

详细报告见 [Lexmount Browser](lexbrowser_results.md) 与 [Local Chrome](local_results.md)。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _campaign_timestamps(campaign: str) -> dict[str, str]:
    if not re.fullmatch(r"\d{8}_\d{6}", campaign):
        raise ValueError("campaign must use YYYYMMDD_HHMMSS")
    base = dt.datetime.strptime(campaign, "%Y%m%d_%H%M%S")
    return {
        "lexmount": base.strftime("%Y%m%d_%H%M%S"),
        "local": (base + dt.timedelta(seconds=1)).strftime("%Y%m%d_%H%M%S"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=pathlib.Path, default=pathlib.Path("/data/wf/sxh"))
    parser.add_argument(
        "--campaign",
        "--campaign-id",
        dest="campaign",
        default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--backend", choices=["lexmount", "local", "all"], default="all")
    parser.add_argument(
        "--stage", choices=["prepare", "rollout", "judge", "report", "all"], default="all"
    )
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--max-rollout-passes", type=int, default=3)
    parser.add_argument("--policy-label", default="Qwen3-8B")
    parser.add_argument("--policy-artifact", type=pathlib.Path)
    parser.add_argument("--policy-sha256", default="")
    parser.add_argument("--rollout-concurrency", type=int, default=ROLLOUT_CONCURRENCY)
    parser.add_argument(
        "--task-count",
        type=int,
        default=TASK_COUNT,
        help="Use a deterministic first_n slice after validating the full pinned 300-task blob.",
    )
    parser.add_argument(
        "--allow-partial-judge",
        action="store_true",
        help="Judge the strict valid subset only; never enables formal report publication",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.task_count <= TASK_COUNT:
        parser.error(f"--task-count must be in 1..{TASK_COUNT}")
    if args.rollout_concurrency < 1:
        parser.error("--rollout-concurrency must be positive")

    policy_model = resolve_policy_metadata(args)
    env_values = validate_environment(os.environ)
    policy_model["endpoint_model_id"] = env_values["QWEN_MODEL_ID"]
    runtime = args.runtime_root.resolve()
    root = runtime / ".online_mind2web_v2"
    checkout = root / f"browseruse-agent-bench-{BUBENCH_COMMIT[:12]}"
    osu_checkout = root / f"Online-Mind2Web-{OSU_COMMIT[:12]}"
    config = root / "config.yaml"
    campaign_dir = runtime / "results/online_mind2web_v2" / args.campaign
    reports_dir = campaign_dir / "reports"
    backends = list(BACKENDS) if args.backend == "all" else [args.backend]
    timestamps = _campaign_timestamps(args.campaign)

    # Fail fast before cloning or installing anything.  A stale or merely
    # similar dataset is never accepted as a substitute for the pinned blob.
    all_tasks, dataset_manifest = load_pinned_dataset(os.environ)
    tasks = all_tasks[: args.task_count]
    dataset_manifest = {
        **dataset_manifest,
        "full_task_count": len(all_tasks),
        "selected_task_count": len(tasks),
        "selection": {"kind": "first_n", "count": len(tasks)},
        "selected_task_ids_sha256": hashlib.sha256(
            "\n".join(task["task_id"] for task in tasks).encode("utf-8")
        ).hexdigest(),
    }
    verify_api_model(env_values["QWEN_BASE_URL"], env_values["QWEN_API_KEY"], MODEL_ID)
    verify_api_model(env_values["JUDGE_BASE_URL"], env_values["JUDGE_API_KEY"], JUDGE_MODEL)

    bootstrap_checkout(BUBENCH_REPO, BUBENCH_COMMIT, checkout)
    bootstrap_checkout(
        OSU_REPO,
        OSU_COMMIT,
        osu_checkout,
        sparse_paths=("src", "script"),
    )
    adapter = apply_v2_recording_adapter(checkout)
    install_bubench(checkout, args.uv)
    judge_python = install_osu_judge(osu_checkout, args.uv)
    official_osu_integrity = osu_integrity(osu_checkout)
    judge_checkout = osu_checkout
    judge_source_patch: dict[str, Any] = {
        "kind": f"Judge source not prepared for stage {args.stage}",
    }
    if args.stage in {"judge", "all"}:
        judge_checkout, judge_source_patch = prepare_judge_source(
            osu_checkout,
            campaign_dir,
            source_tag=args.backend,
        )
    task_file = write_dataset_for_bubench(checkout, tasks)
    write_config(config, checkout)
    manifest = {
        "created_at": utc_now(),
        "campaign": args.campaign,
        "bubench_commit": BUBENCH_COMMIT,
        "osu_commit": OSU_COMMIT,
        "dataset": dataset_manifest,
        "policy_model": policy_model,
        "recording_adapter": adapter,
        "osu_integrity": official_osu_integrity,
        "judge_source_patch": judge_source_patch,
        "task_file_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
        "selected_task_count": len(tasks),
        "rollout_concurrency": args.rollout_concurrency,
        "judge": {
            "model": JUDGE_MODEL,
            "concurrency": JUDGE_CONCURRENCY,
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS,
            "max_tries": JUDGE_MAX_TRIES,
            "score_threshold": JUDGE_SCORE_THRESHOLD,
        },
        "timestamps": timestamps,
    }
    atomic_json(campaign_dir / "manifest.json", manifest)
    if args.stage == "prepare":
        log("prepare complete")
        return 0

    states: dict[str, tuple[pathlib.Path, dict[str, Any], dict[str, Any] | None]] = {}
    for backend in backends:
        timestamp = timestamps[backend]
        run_dir = official_run_dir(checkout, timestamp)
        # Bubench treats an explicit --timestamp as a resume binding and
        # requires the directory to exist.  Pre-create the empty formal run
        # directory so the fixed timestamp is authoritative from task one.
        run_dir.mkdir(parents=True, exist_ok=True)
        rollout = normalize_run(run_dir, tasks)
        if args.stage in {"rollout", "all"} and not rollout["complete"]:
            for attempt in range(1, args.max_rollout_passes + 1):
                log(
                    f"{backend}: rollout pass {attempt}, valid={rollout['valid_count']}/{len(tasks)}"
                )
                result = run(
                    build_rollout_command(
                        checkout,
                        config,
                        backend,
                        timestamp,
                        args.rollout_concurrency,
                    ),
                    cwd=checkout,
                    env=os.environ,
                    check=False,
                )
                rollout = normalize_run(run_dir, tasks)
                atomic_json(campaign_dir / backend / "rollout_audit.json", rollout)
                if rollout["complete"]:
                    break
                log(f"{backend}: bubench exit={result.returncode}; resume remains incomplete")
        atomic_json(campaign_dir / backend / "rollout_audit.json", rollout)
        if args.stage == "rollout":
            states[backend] = (run_dir, rollout, None)
            continue
        partial_judge = (
            args.stage == "judge" and args.allow_partial_judge and bool(rollout["valid_task_ids"])
        )
        if not rollout["complete"] and not partial_judge:
            raise RuntimeError(
                f"{backend} rollout incomplete: {rollout['valid_count']}/{len(tasks)}; "
                "Judge and report were not started"
            )

        judge_dir = campaign_dir / backend / "official_webjudge"
        judge_file = judge_output_path(judge_dir)
        judge = inspect_judge(judge_file, set(rollout["valid_task_ids"]))
        if args.stage in {"judge", "all"} and not judge["complete"]:
            judge_dir.mkdir(parents=True, exist_ok=True)
            judge_env = dict(os.environ)
            judge_env["OPENAI_API_KEY"] = env_values["JUDGE_API_KEY"]
            judge_env["OPENAI_BASE_URL"] = env_values["JUDGE_BASE_URL"]
            judge = run_official_judge_isolated(
                judge_checkout,
                judge_python,
                run_dir / "tasks",
                judge_dir,
                set(rollout["valid_task_ids"]),
                env_values["JUDGE_API_KEY"],
                judge_env,
            )
        atomic_json(campaign_dir / backend / "judge_audit.json", judge)
        states[backend] = (run_dir, rollout, judge)

    if args.stage in {"all", "report"}:
        for backend, (run_dir, rollout, judge) in states.items():
            if judge is None:
                judge_dir = campaign_dir / backend / "official_webjudge"
                judge = inspect_judge(judge_output_path(judge_dir), set(rollout["valid_task_ids"]))
            report_name = "lexbrowser_results.md" if backend == "lexmount" else "local_results.md"
            write_report(
                reports_dir / report_name,
                backend,
                args.campaign,
                dataset_manifest,
                rollout,
                judge,
                run_dir,
                policy_model,
                args.rollout_concurrency,
                len(tasks),
            )
        write_comparison_report(
            reports_dir / "README.md",
            args.campaign,
            states,
            policy_model,
            args.rollout_concurrency,
            len(tasks),
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        log(f"FATAL: {exc}")
        raise SystemExit(2) from None
