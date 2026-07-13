import pathlib
import tempfile
import unittest

from lexbrowser_eval.lexbench.qwen3_8b import campaign as MODULE


class StressCampaignTests(unittest.TestCase):
    def test_runtime_env_parser_returns_values_without_export_syntax(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / ".runtime.env"
            path.write_text(
                "# ignored\nexport QWEN_BASE_URL='http://127.0.0.1:18088/v1'\n"
                "QWEN_MODEL_ID=qwen3_8B\n",
                encoding="utf-8",
            )

            values = MODULE.read_runtime_env(path)

        self.assertEqual(values["QWEN_BASE_URL"], "http://127.0.0.1:18088/v1")
        self.assertEqual(values["QWEN_MODEL_ID"], "qwen3_8B")

    def test_qwen_metrics_url_removes_only_v1_suffix(self):
        self.assertEqual(
            MODULE.qwen_metrics_url("http://127.0.0.1:18088/v1"),
            "http://127.0.0.1:18088/metrics",
        )

    def test_capacity_probe_uses_complete_twenty_task_replicas(self):
        timestamps = [f"20260710_1200{second:02d}" for second in range(4)]

        manifest = MODULE.build_probe_manifest("local", 80, timestamps)

        self.assertTrue(manifest["capacity_probe"])
        self.assertEqual(manifest["replica_count"], 4)
        self.assertEqual(len(manifest["task_ids"]), 20)
        with self.assertRaises(ValueError):
            MODULE.build_probe_manifest("local", 70, timestamps)

    def test_binary_search_chooses_a_twenty_aligned_midpoint(self):
        self.assertEqual(MODULE.binary_targets(60, 100), [80])
        self.assertEqual(MODULE.binary_targets(200, 500), [340])
        self.assertEqual(MODULE.binary_targets(80, 100), [])


if __name__ == "__main__":
    unittest.main()
