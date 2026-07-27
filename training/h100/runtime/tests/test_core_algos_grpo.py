"""Unit tests for the patched GRPO advantage estimator.

Runs against ``runtime/patches/core_algos.py`` without importing verl: the
target function is extracted from the patch file source, so the only
dependencies are torch and numpy.

    python3 training/h100/runtime/tests/test_core_algos_grpo.py
"""

import os
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PATCH_FILE = Path(__file__).resolve().parents[1] / "patches" / "core_algos.py"


def load_grpo_fn():
    source = PATCH_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"def compute_grpo_outcome_advantage\(.*?\n    return scores, scores",
        source,
        re.S,
    )
    body = match.group(0).replace("config: Optional[AlgoConfig] = None,", "config=None,")
    body = body.replace("env_invalid: Optional[torch.Tensor] = None,", "env_invalid=None,")
    namespace = {"torch": torch, "np": np, "os": os, "defaultdict": defaultdict, "Any": object}
    exec(body, namespace)
    return namespace["compute_grpo_outcome_advantage"]


GRPO = load_grpo_fn()
GROUP = np.array(["g"] * 8)
RESPONSE_LEN = 4


def make_batch(scores, zero_mask_rows=(), env_invalid=None):
    rewards = torch.zeros(len(scores), RESPONSE_LEN)
    for i, s in enumerate(scores):
        rewards[i, -1] = s
    mask = torch.ones(len(scores), RESPONSE_LEN, dtype=torch.long)
    for i in zero_mask_rows:
        mask[i] = 0
    flags = None
    if env_invalid is not None:
        flags = torch.tensor(env_invalid, dtype=torch.bool)
    return rewards, mask, flags


def per_sample(adv, row):
    return float(adv[row].abs().max())


class ExplicitFlagTest(unittest.TestCase):
    def test_env_invalid_excluded_from_stats(self):
        """2/8 env failures (under the breaker): stats over the 6 valid only.

        Note the reviewer's original 4/8 example now falls under the 25%
        group breaker (0.5 > 0.25) and is dropped whole — covered by
        test_env_invalid_fraction_trips_breaker.
        """
        rewards, mask, flags = make_batch(
            [1, 1, 0, 0, 0, 0, 0, 0],
            zero_mask_rows=(6, 7),
            env_invalid=[False] * 6 + [True] * 2,
        )
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        # valid group [1,1,0,0,0,0]: adv(+) = (1-1/3)/std = 1.291
        self.assertAlmostEqual(per_sample(adv, 0), 1.291, places=3)
        self.assertAlmostEqual(per_sample(adv, 2), 0.645, places=3)
        for i in (6, 7):
            self.assertEqual(per_sample(adv, i), 0.0)

    def test_policy_empty_participates_in_baseline(self):
        """Source (b): zero-token policy stop; env_invalid=False.

        Its reward=0 must enter the group mean/std; its own advantage stays 0
        because every response token is loss-masked; it must not count toward
        the invalid-fraction circuit breaker.
        """
        rewards, mask, flags = make_batch(
            [1, 1, 0, 0, 0, 0, 0, 0],
            zero_mask_rows=(5, 6, 7),   # three policy-empty rows
            env_invalid=[False] * 8,     # none of them environment failures
        )
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        # Baseline over ALL 8 scores [1,1,0,0,0,0,0,0]: adv(+) = 1.620
        self.assertAlmostEqual(per_sample(adv, 0), 1.620, places=3)
        # policy-empty rows carry no gradient (mask multiply)
        for i in (5, 6, 7):
            self.assertEqual(per_sample(adv, i), 0.0)
        # true failures with real tokens keep their negative advantage
        self.assertAlmostEqual(per_sample(adv, 2), 0.540, places=3)

    def test_mixed_policy_empties_do_not_trip_breaker(self):
        """Regression for the review's amplification case: 3x source (b) +
        5 normal must keep the group (previously mask inference dropped it)."""
        rewards, mask, flags = make_batch(
            [1, 1, 0, 0, 0, 0, 0, 0],
            zero_mask_rows=(5, 6, 7),
            env_invalid=[False] * 8,
        )
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        self.assertGreater(adv.abs().sum().item(), 0.0)

    def test_env_invalid_fraction_trips_breaker(self):
        """3/8 env-invalid = 0.375 > 0.25: whole group dropped."""
        rewards, mask, flags = make_batch(
            [1, 1, 0, 0, 0, 0, 0, 0],
            zero_mask_rows=(5, 6, 7),
            env_invalid=[False] * 5 + [True] * 3,
        )
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        self.assertEqual(adv.abs().sum().item(), 0.0)

    def test_env_invalid_fraction_boundary_kept(self):
        """2/8 env-invalid = 0.25 is not > 0.25: group kept, stats over 6."""
        rewards, mask, flags = make_batch(
            [1, 1, 0, 0, 0, 0, 0, 0],
            zero_mask_rows=(6, 7),
            env_invalid=[False] * 6 + [True] * 2,
        )
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        expected = (1 - 2 / 6) / (torch.tensor([1.0, 1, 0, 0, 0, 0]).std() + 1e-6)
        self.assertAlmostEqual(per_sample(adv, 0), float(expected), places=3)


class FallbackTest(unittest.TestCase):
    def test_none_flags_fall_back_to_mask_inference(self):
        """env_invalid=None reproduces the pre-fix mask-heuristic behavior:
        zero-mask rows are treated as invalid (2/8 stays under the breaker)."""
        rewards, mask, _ = make_batch([1, 1, 0, 0, 0, 0, 0, 0], zero_mask_rows=(6, 7))
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=None)
        self.assertAlmostEqual(per_sample(adv, 0), 1.291, places=3)

    def test_all_valid_matches_upstream_formula(self):
        """No invalid samples: identical to the unpatched estimator."""
        rewards, mask, flags = make_batch([1, 1, 0, 0, 0, 0, 0, 0], env_invalid=[False] * 8)
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        scores = torch.tensor([1.0, 1, 0, 0, 0, 0, 0, 0])
        expected = (1 - scores.mean()) / (scores.std() + 1e-6)
        self.assertAlmostEqual(per_sample(adv, 0), float(expected), places=3)

    def test_min_valid_per_group_drop(self):
        """Only one valid sample left: baseline meaningless, group dropped."""
        rewards, mask, flags = make_batch(
            [1, 0, 0, 0, 0, 0, 0, 0],
            zero_mask_rows=range(1, 8),
            env_invalid=[False] + [True] * 7,
        )
        adv, _ = GRPO(rewards, mask, GROUP, env_invalid=flags)
        self.assertEqual(adv.abs().sum().item(), 0.0)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
