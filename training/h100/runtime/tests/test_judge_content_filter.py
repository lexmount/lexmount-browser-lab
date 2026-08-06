"""Provider moderation of the judge prompt is its own failure class.

On 2026-08-05 the judge endpoint (DMX / deepseek-v4-flash) started returning
HTTP 400 vendor code 10013 for WebVoyager's BBC News tasks: the judge prompt
embeds the page text, so war/politics coverage tripped the upstream content
filter.  It was gone again on 08-06.  Retrying the same model reproduces it
exactly, so the judge must fail fast and hand the prompt to a fallback.
"""
import sys
import types
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1]

FILTER_400 = (
    "Error code: 400 - {'error': {'message': '非常抱歉，根据相关法律法规，我们无法提供关于"
    "以下内容的答案', 'type': 'invalid_request_error', 'param': '', 'code': 10013}}"
)


def _load_server_helpers():
    """Import the judge helpers without fastapi/openai installed."""
    for name in ("fastapi", "pydantic", "uvicorn", "openai", "nemo_gym"):
        sys.modules.setdefault(name, types.ModuleType(name))
    source = (RUNTIME / "nemo_gym_webvoyager_server.py").read_text(encoding="utf-8")
    start = source.index("_CONTENT_FILTER_MARKERS =")
    end = source.index("def _extract_structured_judge_result", start)
    module = types.ModuleType("_judge_helpers")
    exec(compile(source[start:end], "nemo_gym_webvoyager_server.py", "exec"), module.__dict__)
    return module._is_content_filter_refusal


def _load_agent_helper():
    for name in ("verl", "verl.experimental", "verl.experimental.agent_loop",
                 "verl.tools", "verl.tools.base_tool", "verl.tools.schemas",
                 "verl.experimental.agent_loop.tool_agent_loop"):
        sys.modules.setdefault(name, types.ModuleType(name))
    source = (RUNTIME / "lexbrowser_verl_agent.py").read_text(encoding="utf-8")
    start = source.index("_INFRASTRUCTURE_ERROR_PREFIX =")
    end = source.index("class BrowserTool", start)
    module = types.ModuleType("_classifier")
    module.__dict__["Any"] = object
    exec(compile(source[start:end], "lexbrowser_verl_agent.py", "exec"), module.__dict__)
    return module._judge_failure_reason


is_content_filter = _load_server_helpers()
judge_reason = _load_agent_helper()


class _ApiError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


# --- detecting the refusal --------------------------------------------------

def test_vendor_code_10013_is_a_content_filter():
    assert is_content_filter(_ApiError(FILTER_400)) is True


def test_detection_works_without_a_status_code_attribute():
    # The SDK surfaces some errors as plain exceptions carrying only the text.
    assert is_content_filter(Exception(f"BadRequestError: {FILTER_400}")) is True


def test_rate_limit_is_not_a_content_filter():
    # 429 must keep retrying — it is transient, unlike moderation.
    assert is_content_filter(_ApiError("Error code: 429 - rate limited", 429)) is False


def test_timeout_is_not_a_content_filter():
    assert is_content_filter(TimeoutError("Request timed out")) is False


def test_unrelated_400_is_not_a_content_filter():
    assert is_content_filter(
        _ApiError("Error code: 400 - {'error': {'message': 'max_tokens too large'}}")
    ) is False


def test_server_error_is_not_a_content_filter():
    assert is_content_filter(_ApiError("Error code: 500 - upstream failure", 500)) is False


# --- reporting it apart from a bad verdict ----------------------------------

def test_content_filter_gets_its_own_invalid_reason():
    assert judge_reason(
        {"lexbrowser_reason": "judge_error:JudgeContentFilter"}
    ) == "environment_judge_content_filtered"


def test_unparseable_verdict_stays_the_generic_judge_failure():
    assert judge_reason(
        {"lexbrowser_reason": "judge_error:InvalidJudgeResponse"}
    ) == "environment_judge_failed"


def test_missing_reason_defaults_to_the_generic_judge_failure():
    assert judge_reason({}) == "environment_judge_failed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
