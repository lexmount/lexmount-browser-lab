import pathlib
import unittest

from lexbrowser_eval.lexbench.qwen3_8b.protocol import (
    PROTOCOL,
    STRESS_SCHEDULE,
    resolve_runtime_paths,
)


class LexBenchConfigTests(unittest.TestCase):
    def test_revised_protocol_is_frozen(self):
        self.assertEqual(PROTOCOL.agent_model_id, "qwen3_8B")
        self.assertEqual(PROTOCOL.model_config_name, "qwen3-8B")
        self.assertEqual(PROTOCOL.quality_split, "All")
        self.assertEqual(PROTOCOL.quality_task_count, 210)
        self.assertEqual(PROTOCOL.quality_concurrency, 10)
        self.assertEqual(PROTOCOL.judge_model, "gpt-5.4")
        self.assertEqual(PROTOCOL.judge_strategy, "stepwise")
        self.assertEqual(
            STRESS_SCHEDULE,
            (
                (20, 20, "lexmount"),
                (20, 20, "local"),
                (50, 50, "local"),
                (50, 50, "lexmount"),
            ),
        )

    def test_runtime_paths_use_data_root(self):
        paths = resolve_runtime_paths(pathlib.Path("/repo"), pathlib.Path("/data/wf/sxh"))
        self.assertEqual(
            paths.checkout, pathlib.Path("/data/wf/sxh/.lexbench/browseruse-agent-bench")
        )
        self.assertEqual(paths.results_root, pathlib.Path("/data/wf/sxh/results/lexbench"))


if __name__ == "__main__":
    unittest.main()
