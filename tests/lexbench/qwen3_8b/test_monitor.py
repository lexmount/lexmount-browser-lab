import pathlib
import tempfile
import unittest

from lexbrowser_eval.lexbench.qwen3_8b.monitor import (
    classify_local_browser,
    discover_new_run_dir,
    parse_args,
    parse_nvidia_rows,
    phase_for,
)


class NvidiaParsingTests(unittest.TestCase):
    def test_parse_nvidia_rows_keeps_each_gpu(self):
        rows = parse_nvidia_rows(
            "0, GPU-a, 91, 62, 19876, 32607, 245.5\n1, GPU-b, 88, 59, 27928, 32607, 231.0\n"
        )

        self.assertEqual(
            rows,
            [
                {
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-a",
                    "gpu_sm_percent": 91.0,
                    "gpu_memory_percent": 62.0,
                    "gpu_memory_used_mib": 19876.0,
                    "gpu_memory_total_mib": 32607.0,
                    "gpu_power_w": 245.5,
                },
                {
                    "gpu_index": 1,
                    "gpu_uuid": "GPU-b",
                    "gpu_sm_percent": 88.0,
                    "gpu_memory_percent": 59.0,
                    "gpu_memory_used_mib": 27928.0,
                    "gpu_memory_total_mib": 32607.0,
                    "gpu_power_w": 231.0,
                },
            ],
        )

    def test_parse_nvidia_rows_rejects_malformed_output(self):
        with self.assertRaisesRegex(RuntimeError, "NVIDIA sample"):
            parse_nvidia_rows("0, missing-fields\n")


class RunDiscoveryTests(unittest.TestCase):
    def test_discovers_exactly_one_new_timestamp_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "old").mkdir()
            before = {"old"}
            (root / "new").mkdir()

            self.assertEqual(discover_new_run_dir(root, before), root / "new")

    def test_rejects_ambiguous_new_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "one").mkdir()
            (root / "two").mkdir()

            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                discover_new_run_dir(root, set())

    def test_cli_accepts_existing_official_run_for_resume(self):
        args = parse_args(
            [
                "--output-dir",
                "/out",
                "--cwd",
                "/checkout",
                "--experiment-root",
                "/experiments",
                "--existing-run-dir",
                "/experiments/20260710_162153",
                "--expected-tasks",
                "210",
                "--",
                "uv",
                "run",
                "bubench",
                "run-eval",
            ]
        )

        self.assertEqual(
            args.existing_run_dir,
            pathlib.Path("/experiments/20260710_162153"),
        )


class ClassificationTests(unittest.TestCase):
    def test_local_chrome_is_classified_but_cdp_client_is_not(self):
        self.assertTrue(
            classify_local_browser("chrome", "/usr/bin/google-chrome --remote-debugging-port=0")
        )
        self.assertTrue(classify_local_browser("chromium", "/usr/bin/chromium"))
        self.assertFalse(classify_local_browser("python", "browser-use cdp client"))

    def test_judge_phase_starts_only_after_all_rollout_results(self):
        self.assertEqual(phase_for(False, 20, 20), "rollout")
        self.assertEqual(phase_for(True, 19, 20), "rollout")
        self.assertEqual(phase_for(True, 20, 20), "judge")


if __name__ == "__main__":
    unittest.main()
