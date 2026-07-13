import pathlib
import tempfile
import unittest

from lexbrowser_eval.lexbench.cli import parse_args, quality_command, selected_backends


class LexBenchRunnerCliTests(unittest.TestCase):
    def test_qwen_defaults_run_both_backends_and_all_modes(self):
        args = parse_args(["qwen3-8b", "--env-file", "/tmp/eval.env"])
        self.assertEqual(args.backend, "all")
        self.assertEqual(args.mode, "all")
        self.assertEqual(args.stage, "all")

    def test_model_override_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "qwen3-8b",
                    "--env-file",
                    "/tmp/eval.env",
                    "--model",
                    "glm-5.2",
                ]
            )

    def test_backend_selection_is_explicit(self):
        self.assertEqual(selected_backends("all"), ("lexmount", "local"))
        self.assertEqual(selected_backends("local"), ("local",))

    def test_quality_rollout_command_uses_official_skip_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            command, marker, existing = quality_command(
                checkout=root / "checkout",
                results_root=root / "results",
                campaign_id="campaign",
                backend="lexmount",
                task_count=10,
                rollout_only=True,
                resume=False,
            )
        self.assertIn("run-eval", command)
        self.assertIn("--skip-eval", command)
        self.assertIn("--concurrency", command)
        self.assertIn("10", command)
        self.assertIn("first_n", command)
        self.assertIn("--count", command)
        self.assertEqual(marker.name, "official_run_dir.txt")
        self.assertIsNone(existing)

    def test_task_count_defaults_to_official_all(self):
        args = parse_args(["qwen3-8b", "--env-file", "/tmp/eval.env"])
        self.assertEqual(args.task_count, 210)


if __name__ == "__main__":
    unittest.main()
