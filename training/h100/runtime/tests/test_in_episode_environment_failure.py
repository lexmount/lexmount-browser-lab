"""In-episode environment failures must not reach the GRPO group statistics.

The boundary failures (reset, unknown session, close RPC, judge) were already
classified.  These cover the mid-episode case: a browser that breaks while the
policy is acting, which the lightmount backend surfaces as a hung CDP call.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1]


def _load_classifier():
    """Import the classifier without verl/nemo-gym installed."""
    for name in ("verl", "verl.experimental", "verl.experimental.agent_loop",
                 "verl.tools", "verl.tools.base_tool", "verl.tools.schemas",
                 "verl.experimental.agent_loop.tool_agent_loop"):
        sys.modules.setdefault(name, types.ModuleType(name))
    source = (RUNTIME / "lexbrowser_verl_agent.py").read_text(encoding="utf-8")
    start = source.index("def _in_episode_environment_failure")
    end = source.index("class BrowserTool", start)
    module = types.ModuleType("_classifier")
    module.__dict__["Any"] = object
    exec(compile(source[start:end], "lexbrowser_verl_agent.py", "exec"), module.__dict__)
    return module._in_episode_environment_failure


classify = _load_classifier()


def test_breaker_tripped_by_browser_is_invalid():
    # act() timed out (lightmount hangs the CDP connection after a click on
    # some sites), the guard tripped, the judge then said "no".
    assert classify(0.0, {
        "termination_reason": "infrastructure_act_timeout",
        "infrastructure_tool_failures": 1,
        "final_answer_present": False,
    }) == "environment_infrastructure_act_timeout"


def test_episode_timeout_is_invalid():
    assert classify(0.0, {
        "termination_reason": "infrastructure_episode_timeout",
        "infrastructure_tool_failures": 1,
        "final_answer_present": False,
    }) == "environment_infrastructure_episode_timeout"


def test_policy_failure_stays_in_the_batch():
    # No infrastructure failure at all: an honest reward=0 the policy earned.
    assert classify(0.0, {
        "termination_reason": "",
        "infrastructure_tool_failures": 0,
        "final_answer_present": True,
    }) == ""


def test_policy_no_progress_breaker_stays_in_the_batch():
    # The breaker tripped on the *policy* repeating itself — real signal.
    assert classify(0.0, {
        "termination_reason": "policy_no_progress_repeated_tool_call",
        "infrastructure_tool_failures": 0,
        "final_answer_present": False,
    }) == ""


def test_successful_rollout_is_never_invalidated():
    # A transient browser error the policy worked around anyway: keep it, or
    # the group mean is biased downward by dropping only the wins.
    assert classify(1.0, {
        "termination_reason": "infrastructure_act_timeout",
        "infrastructure_tool_failures": 2,
        "final_answer_present": True,
    }) == ""


def test_transient_failure_without_answer_is_invalid():
    # Breaker never tripped, but the browser failed and the policy never got to
    # answer — not a fair trial.
    assert classify(0.0, {
        "termination_reason": "",
        "infrastructure_tool_failures": 2,
        "final_answer_present": False,
    }) == "environment_tool_infrastructure_failure"


def test_transient_failure_with_answer_stays_in_the_batch():
    assert classify(0.0, {
        "termination_reason": "",
        "infrastructure_tool_failures": 1,
        "final_answer_present": True,
    }) == ""


def test_missing_fields_default_to_valid():
    # Older environment services do not publish the guard counters; absence
    # must not silently invalidate every rollout.
    assert classify(0.0, {}) == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
