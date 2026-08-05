"""In-episode environment failures must not reach the GRPO group statistics.

The boundary failures (reset, unknown session, close RPC, judge) were already
classified.  These cover the mid-episode case, and in particular that the test
is *causal*: a browser error the policy recovered from leaves the outcome the
policy's own, and must stay in the batch as real negative signal.
"""
import sys
import types
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1]


def _load():
    """Import the classifier helpers without verl/nemo-gym installed."""
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
    return module._in_episode_environment_failure, module._ends_on_infrastructure_failure


classify, ends_on_infra = _load()


# --- the causal test itself -------------------------------------------------

def test_breaker_tripped_by_browser_is_invalid():
    assert classify(0.0, {
        "termination_reason": "infrastructure_act_timeout",
        "trailing_infrastructure_failure": True,
        "final_answer_present": False,
    }) == "environment_infrastructure_act_timeout"


def test_episode_timeout_is_invalid():
    assert classify(0.0, {
        "termination_reason": "infrastructure_episode_timeout",
        "trailing_infrastructure_failure": False,
        "final_answer_present": False,
    }) == "environment_infrastructure_episode_timeout"


def test_ends_on_broken_call_without_answer_is_invalid():
    # The browser broke and the policy never got another working call.
    assert classify(0.0, {
        "termination_reason": "",
        "trailing_infrastructure_failure": True,
        "final_answer_present": False,
    }) == "environment_tool_infrastructure_failure"


def test_recovered_transient_failure_stays_in_the_batch():
    # This is the regression the causal test exists for: the browser failed
    # mid-episode, the policy recovered and kept working, and then failed on
    # its own merits.  Discarding it would throw away real negative signal and
    # inflate the measured environment failure rate.
    assert classify(0.0, {
        "termination_reason": "",
        "trailing_infrastructure_failure": False,
        "infrastructure_tool_failures": 3,
        "final_answer_present": False,
    }) == ""


def test_broken_last_call_but_answer_present_stays():
    assert classify(0.0, {
        "termination_reason": "",
        "trailing_infrastructure_failure": True,
        "final_answer_present": True,
    }) == ""


def test_policy_failure_stays_in_the_batch():
    assert classify(0.0, {
        "termination_reason": "",
        "trailing_infrastructure_failure": False,
        "final_answer_present": True,
    }) == ""


def test_policy_no_progress_breaker_stays_in_the_batch():
    assert classify(0.0, {
        "termination_reason": "policy_no_progress_repeated_tool_call",
        "trailing_infrastructure_failure": False,
        "final_answer_present": False,
    }) == ""


def test_successful_rollout_is_never_invalidated():
    assert classify(1.0, {
        "termination_reason": "infrastructure_act_timeout",
        "trailing_infrastructure_failure": True,
        "final_answer_present": True,
    }) == ""


def test_missing_fields_default_to_valid():
    assert classify(0.0, {}) == ""


# --- trailing-failure detection --------------------------------------------

def test_trailing_detection_uses_the_last_call():
    assert ends_on_infra([
        "ERROR_INFRASTRUCTURE_ACT: CDP failed",
        "Success: clicked",
    ]) is False
    assert ends_on_infra([
        "Success: clicked",
        "ERROR_INFRASTRUCTURE_OBSERVE: CDP failed",
    ]) is True


def test_policy_errors_are_not_infrastructure():
    # A stale selector is the policy's grounding mistake; the browser is fine.
    assert ends_on_infra(["ERROR_POLICY_ACT: selector not found"]) is False


def test_blank_results_do_not_hide_the_last_real_call():
    assert ends_on_infra([
        "ERROR_INFRASTRUCTURE_ACT: CDP failed",
        "   ",
        "",
    ]) is True


def test_empty_episode_is_not_an_infrastructure_failure():
    assert ends_on_infra([]) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
