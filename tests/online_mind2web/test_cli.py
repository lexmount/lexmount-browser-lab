import json
import pathlib
import tempfile
import unittest
from unittest import mock

from lexbrowser_eval.online_mind2web import cli as runner


class OnlineMind2WebV2Tests(unittest.TestCase):
    def setUp(self):
        self.runner = runner

    def official_task(self, task_id="task-a"):
        return {
            "task_id": task_id,
            "confirmed_task": "Find the official result.",
            "website": "https://example.com/",
            "reference_length": 4,
            "level": "easy",
        }

    def upstream_result(self, task_dir, *, answer="done", agent_done="done"):
        task_dir.mkdir(parents=True)
        (task_dir / "trajectory").mkdir()
        (task_dir / "trajectory" / "screenshot-1.png").write_bytes(b"png-one")
        (task_dir / "api_logs").mkdir()
        step = {
            "metadata": {"task_id": task_dir.name, "step_number": 1},
            "input": {"url": "https://example.com/"},
            "output": {
                "thinking": "The requested result is visible.",
                "next_goal": "Finish.",
                "actions": [{"done": {"text": answer, "success": True}}],
            },
            "action_results": [{"error": None, "is_done": True}],
        }
        (task_dir / "api_logs" / "step_001.json").write_text(json.dumps(step))
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "task_id": task_dir.name,
                    "task": "augmented runtime prompt that must not reach Judge",
                    "answer": answer,
                    "agent_done": agent_done,
                    "env_status": "success",
                    "browser_id": "lexmount",
                    "action_history": ["legacy upstream action"],
                }
            )
        )

    def test_fixed_scope_and_concurrency(self):
        self.assertEqual(self.runner.TASK_COUNT, 300)
        self.assertEqual(self.runner.ROLLOUT_CONCURRENCY, 10)
        self.assertEqual(self.runner.JUDGE_CONCURRENCY, 10)
        self.assertEqual(set(self.runner.BACKENDS), {"lexmount", "local"})

    def test_official_revisions_are_pinned(self):
        self.assertRegex(self.runner.BUBENCH_COMMIT, r"^[0-9a-f]{40}$")
        self.assertRegex(self.runner.OSU_COMMIT, r"^[0-9a-f]{40}$")
        self.assertRegex(self.runner.HF_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(self.runner.HF_GIT_BLOB_OID, r"^[0-9a-f]{40}$")

    def test_git_blob_oid_matches_git_object_format(self):
        # `printf test | git hash-object --stdin`.
        self.assertEqual(
            self.runner.git_blob_oid(b"test"),
            "30d74d258442c7c65512eafab474568dd706c430",
        )

    def test_gated_dataset_fails_closed_without_token_or_exact_file(self):
        with self.assertRaisesRegex(RuntimeError, "gated.*HF_TOKEN"):
            self.runner.load_pinned_dataset({})

    def test_config_freezes_official_agent_and_judge_values(self):
        config = self.runner.CONFIG_TEMPLATE
        self.assertIn("use_vision: false", config)
        self.assertIn("max_steps: 40", config)
        self.assertIn("flash_mode: true", config)
        self.assertIn("timeout: 600", config)
        self.assertIn("temperature: 1", config)
        self.assertIn("max_tries: 3", config)
        self.assertIn("max_tokens: 512", config)
        self.assertNotIn("sk-", config)

    def test_rollout_commands_are_c10_for_both_backends(self):
        for backend, expected in (("lexmount", "lexmount"), ("local", "local")):
            command = self.runner.build_rollout_command(
                pathlib.Path("/checkout"),
                pathlib.Path("/config"),
                backend,
                "20260710_120000",
            )
            self.assertEqual(command[command.index("--concurrency") + 1], "10")
            self.assertEqual(command[command.index("--browser") + 1], expected)
            self.assertIn("--skip-completed", command)
            self.assertEqual(command[command.index("--mode") + 1], "all")
            if backend == "local":
                self.assertEqual(
                    command[:4],
                    [
                        "xvfb-run",
                        "-a",
                        "-s",
                        "-screen 0 1920x1080x24",
                    ],
                )

    def test_judge_command_uses_official_mode_and_one_worker(self):
        command = self.runner.build_judge_command(
            pathlib.Path("/osu"),
            pathlib.Path("/python"),
            pathlib.Path("/tasks"),
            pathlib.Path("/judge"),
            "secret",
        )
        self.assertEqual(command[command.index("--mode") + 1], "WebJudge_Online_Mind2Web_eval")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.4")
        self.assertEqual(command[command.index("--num_worker") + 1], "10")
        self.assertEqual(command[command.index("--score_threshold") + 1], "3")

    def test_judge_temperature_patch_is_isolated_and_narrow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            official = root / "official"
            (official / "src").mkdir(parents=True)
            source = (
                "class OpenaiEngine:\n"
                "    def __init__(self,\n"
                "        temperature=0,\n"
                "        port=-1,\n"
                "    ): pass\n"
                "    def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs):\n"
                "        return messages\n"
            )
            (official / "src/utils.py").write_text(source)
            patched, manifest = self.runner.prepare_judge_source(
                official, root / "campaign", source_tag="local"
            )
            self.assertEqual((official / "src/utils.py").read_text(), source)
            patched_source = (patched / "src/utils.py").read_text()
            self.assertIn("temperature=1", patched_source)
            self.assertIn("max_new_tokens=512", patched_source)
            self.assertEqual(manifest["max_tries"], 3)
            self.assertTrue(str(patched).endswith("temperature_1_local"))

    def test_convert_emits_exact_v2_and_raw_task_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = pathlib.Path(tmp) / "task-a"
            self.upstream_result(task_dir, answer="The result is shown.")
            ok, reason = self.runner.convert_task_result(task_dir, self.official_task())
            self.assertTrue(ok, reason)
            result = json.loads((task_dir / "result.json").read_text())
            self.assertEqual(
                set(result),
                {
                    "schema_version",
                    "task",
                    "task_id",
                    "agent_final_answer",
                    "reference_length",
                    "action_history",
                },
            )
            self.assertEqual(result["schema_version"], "online-mind2web-v2")
            self.assertEqual(result["task"], "Find the official result.")
            self.assertNotIn("runtime prompt", json.dumps(result))
            self.assertEqual(result["agent_final_answer"], "The result is shown.")
            self.assertTrue(result["action_history"][-1]["action"].startswith("TASK_COMPLETE"))
            self.assertTrue((task_dir / "trajectory" / "0000.png").is_file())
            self.assertTrue((task_dir / "bubench_result.json").is_file())

    def test_interrupted_placeholder_is_never_valid_or_skippable(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = pathlib.Path(tmp) / "task-a"
            self.upstream_result(task_dir, answer="Process interrupted")
            ok, reason = self.runner.convert_task_result(task_dir, self.official_task())
            self.assertFalse(ok)
            self.assertIn("interrupted", reason)
            self.assertFalse((task_dir / "result.json").exists())
            self.assertTrue((task_dir / "interrupted_result.json").exists())

    def test_missing_exact_screenshot_rejects_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = pathlib.Path(tmp) / "task-a"
            self.upstream_result(task_dir)
            (task_dir / "trajectory" / "screenshot-1.png").unlink()
            ok, reason = self.runner.convert_task_result(task_dir, self.official_task())
            self.assertFalse(ok)
            self.assertIn("missing exact state screenshot", reason)
            self.assertFalse((task_dir / "result.json").exists())

    def test_non_done_timeout_or_max_steps_is_not_synthesized_or_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = pathlib.Path(tmp) / "task-a"
            self.upstream_result(task_dir, answer="", agent_done="max_steps")
            step_path = task_dir / "api_logs" / "step_001.json"
            step = json.loads(step_path.read_text())
            step["output"]["actions"] = [{"click": {"index": 3}}]
            step["action_results"] = [{"error": None, "is_done": False}]
            step_path.write_text(json.dumps(step))
            raw = json.loads((task_dir / "result.json").read_text())
            raw["error"] = "Maximum steps reached"
            (task_dir / "result.json").write_text(json.dumps(raw))
            ok, reason = self.runner.convert_task_result(task_dir, self.official_task())
            self.assertFalse(ok)
            self.assertIn("no valid TASK_COMPLETE", reason)
            self.assertFalse((task_dir / "result.json").exists())
            self.assertTrue((task_dir / "unterminated_result.json").exists())

    def test_validator_rejects_augmented_task_and_missing_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = pathlib.Path(tmp)
            (task_dir / "trajectory").mkdir()
            (task_dir / "trajectory" / "0000.png").write_bytes(b"png")
            payload = {
                "schema_version": "online-mind2web-v2",
                "task": "augmented",
                "task_id": "task-a",
                "agent_final_answer": None,
                "reference_length": 4,
                "action_history": [
                    {
                        "step": 0,
                        "screenshot": "0000.png",
                        "url": "https://example.com",
                        "action": "WAIT page -> wait",
                        "thought": None,
                        "action_status": None,
                    }
                ],
            }
            errors = self.runner.validate_v2(payload, task_dir, self.official_task())
            self.assertIn("task is not the verbatim official user task", errors)
            self.assertIn("missing terminal TASK_COMPLETE", errors)

    def test_judge_inspection_requires_exact_unique_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "judge.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"task_id": "a", "predicted_label": 1}),
                        json.dumps({"task_id": "b", "predicted_label": 0}),
                    ]
                )
                + "\n"
            )
            audit = self.runner.inspect_judge(path, {"a", "b"})
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["success_rate"], 50.0)
            path.write_text(
                path.read_text() + json.dumps({"task_id": "b", "predicted_label": 0}) + "\n"
            )
            self.assertFalse(self.runner.inspect_judge(path, {"a", "b"})["complete"])

    def test_content_policy_forced_failure_completes_without_forging_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            path = self.runner.judge_output_path(output_dir)
            path.write_text(json.dumps({"task_id": "a", "predicted_label": 1}) + "\n")
            self.runner.atomic_json(
                self.runner.forced_failures_path(output_dir),
                {
                    "b": {
                        "predicted_label": 0,
                        "reason": "content_policy_rejection_after_official_retries",
                    }
                },
            )
            audit = self.runner.inspect_judge(path, {"a", "b"})
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["official_unique_count"], 1)
            self.assertEqual(audit["forced_failure_count"], 1)
            self.assertEqual(audit["successful_tasks"], 1)
            self.assertEqual(audit["failed_tasks"], 1)
            self.assertEqual(audit["success_rate"], 50.0)
            self.assertEqual(len(self.runner.load_judge_records(path)), 1)

    def test_judge_isolates_each_missing_task_while_remaining_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            tasks = root / "tasks"
            output = root / "judge"
            for task_id in ("a", "b"):
                (tasks / task_id).mkdir(parents=True)

            def fake_run(command, **kwargs):
                staging = pathlib.Path(command[command.index("--trajectories_dir") + 1])
                path = self.runner.judge_output_path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    for task_path in sorted(staging.iterdir()):
                        stream.write(
                            json.dumps({"task_id": task_path.name, "predicted_label": 1}) + "\n"
                        )

            with mock.patch.object(self.runner, "run", side_effect=fake_run) as patched:
                audit = self.runner.run_official_judge_isolated(
                    root / "osu",
                    root / "python",
                    tasks,
                    output,
                    {"a", "b"},
                    "secret",
                    {},
                )
            self.assertTrue(audit["complete"])
            self.assertEqual(patched.call_count, 1)
            for call in patched.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--num_worker") + 1], "2")

    def test_report_refuses_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.runner.write_report(
                    pathlib.Path(tmp) / "report.md",
                    "lexmount",
                    "20260710_120000",
                    {"revision": "x", "git_blob_oid": "y", "sha256": "z"},
                    {"complete": False},
                    {"complete": False},
                    pathlib.Path("/run"),
                )

    def test_comparison_report_uses_full_300_task_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "README.md"
            states = {
                "lexmount": (
                    pathlib.Path("/lex"),
                    {"complete": True},
                    {
                        "complete": True,
                        "successful_tasks": 15,
                        "failed_tasks": 285,
                        "success_rate": 5.0,
                        "forced_failure_count": 1,
                    },
                ),
                "local": (
                    pathlib.Path("/local"),
                    {"complete": True},
                    {
                        "complete": True,
                        "successful_tasks": 12,
                        "failed_tasks": 288,
                        "success_rate": 4.0,
                        "forced_failure_count": 2,
                    },
                ),
            }
            self.runner.write_comparison_report(path, "20260711_001343", states)
            report = path.read_text()
            self.assertIn("5.00%（15/300）", report)
            self.assertIn("4.00%（12/300）", report)
            self.assertIn("仅 `WebJudge_Online_Mind2Web_eval`", report)

    def test_timestamp_pair_is_distinct_and_valid(self):
        values = self.runner._campaign_timestamps("20260710_120000")
        self.assertEqual(values["lexmount"], "20260710_120000")
        self.assertEqual(values["local"], "20260710_120001")


if __name__ == "__main__":
    unittest.main()
