"""DAPO dynamic sampling: only groups with reward variance are worth training on.

A GRPO group whose valid rollouts all earn the same reward produces zero
advantage everywhere — and degenerate behaviours (answering without calling
the browser) survive unpunished inside all-zero groups. The norm60d run grew
a 634-episode refusal mode exactly this way. The group filter decides which
groups get committed to TransferQueue and which get resampled.
"""
import sys
import types
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1]


def _load():
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
    return module._group_has_learning_signal


signal = _load()


def test_mixed_rewards_carry_signal():
    assert signal([0.0, 1.0, 0.0, 0.0], [False] * 4) is True


def test_all_zero_group_is_flat():
    # The refusal-mode freeloader case: everyone got 0, nobody gets punished.
    assert signal([0.0] * 8, [False] * 8) is False


def test_all_success_group_is_flat():
    # Solved tasks teach nothing either — same reward everywhere.
    assert signal([1.0] * 8, [False] * 8) is False


def test_invalid_rollouts_cannot_fake_variance():
    # One env-broken rollout at 0 among successes: its reward is an artifact
    # of the environment, not the policy, so the group is still flat.
    assert signal([1.0, 1.0, 1.0, 0.0], [False, False, False, True]) is False


def test_variance_among_valid_rollouts_wins():
    assert signal([1.0, 0.0, 1.0, 0.0], [False, False, True, False]) is True


def test_group_with_one_valid_rollout_has_no_comparison():
    assert signal([1.0, 0.0, 0.0], [False, True, True]) is False


def test_all_invalid_group_gets_resampled():
    assert signal([0.0, 0.0], [True, True]) is False


def test_none_rewards_are_treated_as_invalid():
    # reward_score is None when the judge never produced a verdict.
    assert signal([None, 1.0, 0.0], [False, False, False]) is True
    assert signal([None, None], [False, False]) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
