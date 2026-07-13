from __future__ import annotations

import os
import pathlib
import unittest
from unittest import mock

from lexbrowser_eval.lexbench.qwen3_8b import stage2 as MODULE


class ShardedStage2Tests(unittest.TestCase):
    def test_deterministic_shards_are_balanced_and_complete(self) -> None:
        task_ids = [str(value) for value in range(210, 0, -1)]
        shards = MODULE.deterministic_shards(task_ids, 5)
        self.assertEqual([len(shard) for shard in shards], [42] * 5)
        self.assertEqual(len({task for shard in shards for task in shard}), 210)
        self.assertEqual(shards, MODULE.deterministic_shards(list(reversed(task_ids)), 5))

    def test_duplicate_task_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.deterministic_shards(["1", "1"], 5)

    def test_wrong_judge_model_is_rejected_before_imports(self) -> None:
        values = {
            "LEXBENCH_JUDGE_API_KEY": "test-key",
            "LEXBENCH_JUDGE_BASE_URL": "https://judge.invalid/v1",
            "LEXBENCH_JUDGE_MODEL": "wrong-model",
        }
        with mock.patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(RuntimeError, "must be gpt-5.4"):
                MODULE.judge_settings(pathlib.Path("/unused"))


if __name__ == "__main__":
    unittest.main()
