import json
import pathlib
import subprocess
import sys
import tempfile
import traceback
import unittest
from unittest import mock

from lexbrowser_eval.lexbench.qwen3_8b.official import (
    build_quality_command,
    build_stress_command,
    ensure_checkout,
    freeze_dependencies,
    resolve_output_marker,
    sha256_file,
    sync_official_config,
    validate_checkout,
    validate_environment,
)
from lexbrowser_eval.lexbench.qwen3_8b.protocol import PROTOCOL, required_env_names

EXPECTED_CONFIG = """default:
  agent: browser-use
  data: LexBench-Browser
  model: qwen3-8B
  browser: lexmount
models:
  qwen3-8B:
    model_type: OPENAI
    model_provider: openai
    model_id: $QWEN_MODEL_ID
    api_key: $QWEN_API_KEY
    base_url: $QWEN_BASE_URL
    frequency_penalty: null
    dont_force_structured_output: false
    add_schema_to_system_prompt: true
browsers:
  lexmount:
    browser_id: lexmount
    lexmount_browser_mode: normal
    lexmount_official_proxy: false
    lexmount_api_key: $LEXMOUNT_API_KEY
    lexmount_project_id: $LEXMOUNT_PROJECT_ID
  local:
    browser_id: local
    headless: false
    local_proxy_server: ''
agents:
  browser-use:
    use_judge: false
    use_vision: false
    max_steps: 40
    flash_mode: true
    timeout: 600
site_skills:
  enabled: false
eval:
  model: $LEXBENCH_JUDGE_MODEL
  api_key: $LEXBENCH_JUDGE_API_KEY
  base_url: $LEXBENCH_JUDGE_BASE_URL
  temperature: 1.0
  max_tries: 5
  api_max_images: 50
  detail: high
  max_tokens: 4096
"""


class OfficialCommandTests(unittest.TestCase):
    def test_lexmount_quality_command_has_exact_official_argv(self):
        command = build_quality_command("lexmount", pathlib.Path("/marker"))
        self.assertEqual(
            command,
            [
                "uv",
                "run",
                "bubench",
                "run-eval",
                "--agent",
                "browser-use",
                "--data",
                "LexBench-Browser",
                "--model",
                "qwen3-8B",
                "--browser-id",
                "lexmount",
                "--report-output-dir",
                "/marker",
                "--split",
                "All",
                "--mode",
                "all",
                "--concurrency",
                "10",
                "--eval-strategy",
                "stepwise",
            ],
        )

    def test_lexmount_stress_command_has_exact_official_argv(self):
        command = build_stress_command("lexmount", 20, 20, pathlib.Path("/marker"))
        self.assertEqual(
            command,
            [
                "uv",
                "run",
                "bubench",
                "run-eval",
                "--agent",
                "browser-use",
                "--data",
                "LexBench-Browser",
                "--model",
                "qwen3-8B",
                "--browser-id",
                "lexmount",
                "--report-output-dir",
                "/marker",
                "--split",
                "All",
                "--mode",
                "first_n",
                "--count",
                "20",
                "--no-group-by-site",
                "--concurrency",
                "20",
                "--skip-eval",
            ],
        )
        self.assertNotIn("--eval-strategy", command)
        self.assertNotIn("sample50", command)

    def test_quality_smoke_uses_official_first_n_subset(self):
        command = build_quality_command(
            "lexmount", pathlib.Path("/marker"), task_count=10, concurrency=10
        )
        self.assertIn("first_n", command)
        self.assertEqual(command[command.index("--count") + 1], "10")
        self.assertIn("--no-group-by-site", command)
        self.assertEqual(command[command.index("--eval-strategy") + 1], "stepwise")

    def test_local_quality_command_has_exact_official_argv(self):
        command = build_quality_command("local", pathlib.Path("/marker"))
        self.assertEqual(
            command,
            [
                "xvfb-run",
                "-a",
                "-s",
                "-screen 0 1920x1080x24",
                "uv",
                "run",
                "bubench",
                "run-eval",
                "--agent",
                "browser-use",
                "--data",
                "LexBench-Browser",
                "--model",
                "qwen3-8B",
                "--browser-id",
                "local",
                "--report-output-dir",
                "/marker",
                "--split",
                "All",
                "--mode",
                "all",
                "--concurrency",
                "10",
                "--eval-strategy",
                "stepwise",
            ],
        )

    def test_local_stress_command_has_exact_official_argv(self):
        command = build_stress_command("local", 50, 50, pathlib.Path("/marker"))
        self.assertEqual(
            command,
            [
                "xvfb-run",
                "-a",
                "-s",
                "-screen 0 1920x1080x24",
                "uv",
                "run",
                "bubench",
                "run-eval",
                "--agent",
                "browser-use",
                "--data",
                "LexBench-Browser",
                "--model",
                "qwen3-8B",
                "--browser-id",
                "local",
                "--report-output-dir",
                "/marker",
                "--split",
                "All",
                "--mode",
                "first_n",
                "--count",
                "50",
                "--no-group-by-site",
                "--concurrency",
                "50",
                "--skip-eval",
            ],
        )
        self.assertNotIn("--eval-strategy", command)
        self.assertNotIn("sample50", command)

    def test_stress_command_rejects_cells_outside_frozen_protocol(self):
        with self.assertRaisesRegex(ValueError, "not part of the frozen protocol"):
            build_stress_command("local", 20, 50, pathlib.Path("/marker"))

    def test_config_contains_env_references_not_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            source = checkout / "source.yaml"
            source.write_text("api_key: $QWEN_API_KEY\n", encoding="utf-8")
            target = sync_official_config(checkout, source)
            self.assertEqual(target.read_text(encoding="utf-8"), "api_key: $QWEN_API_KEY\n")

    def test_project_config_has_exact_frozen_secret_free_contract(self):
        source = (
            pathlib.Path(__file__).parents[3] / "experiments" / "qwen3-8b-lexbench" / "config.yaml"
        )
        self.assertEqual(source.read_text(encoding="utf-8"), EXPECTED_CONFIG)

    def test_marker_resolves_relative_to_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            marker = checkout / "marker.txt"
            marker.write_text("experiments/LexBench-Browser/All/run\n", encoding="utf-8")
            self.assertEqual(
                resolve_output_marker(checkout, marker),
                checkout / "experiments/LexBench-Browser/All/run",
            )

    def test_marker_preserves_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            marker = checkout / "marker.txt"
            marker.write_text("/results/lexbench/run\n", encoding="utf-8")
            self.assertEqual(
                resolve_output_marker(checkout, marker),
                pathlib.Path("/results/lexbench/run"),
            )


class OfficialCheckoutTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "file"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.capture_text")
    def test_validate_checkout_rejects_wrong_commit(self, capture_text):
        capture_text.return_value = "wrong\n"
        with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
            validate_checkout(pathlib.Path("/checkout"))
        capture_text.assert_called_once_with(
            ["git", "rev-parse", "HEAD"], pathlib.Path("/checkout")
        )

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.capture_text")
    def test_validate_checkout_rejects_changes_under_official_source(self, capture_text):
        dirty_states = (
            " M browseruse_bench/cli/run.py\n",
            "?? browseruse_bench/cli/untracked.py\n",
        )
        for status in dirty_states:
            with self.subTest(status=status):
                capture_text.side_effect = [PROTOCOL.upstream_commit + "\n", status]
                with self.assertRaisesRegex(RuntimeError, "worktree is not clean"):
                    validate_checkout(pathlib.Path("/checkout"))

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.sha256_file")
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.capture_text")
    def test_validate_checkout_allows_runtime_config_and_pins_all_dataset(
        self, capture_text, sha256_file
    ):
        for status in ("?? config.yaml\n", " M config.yaml\n"):
            with self.subTest(status=status):
                capture_text.side_effect = [PROTOCOL.upstream_commit + "\n", status]
                sha256_file.return_value = PROTOCOL.quality_sha256
                validate_checkout(pathlib.Path("/checkout"))
        self.assertEqual(
            sha256_file.call_args_list,
            [
                mock.call(
                    pathlib.Path("/checkout/browseruse_bench/data/LexBench-Browser/task.jsonl")
                ),
                mock.call(
                    pathlib.Path("/checkout/browseruse_bench/data/LexBench-Browser/task.jsonl")
                ),
            ],
        )

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.sha256_file", return_value="wrong-hash")
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.capture_text")
    def test_validate_checkout_rejects_dataset_hash_mismatch(self, capture_text, _sha256_file):
        capture_text.side_effect = [PROTOCOL.upstream_commit + "\n", ""]
        with self.assertRaisesRegex(RuntimeError, "Dataset hash mismatch"):
            validate_checkout(pathlib.Path("/checkout"))

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.validate_checkout")
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_ensure_checkout_clones_then_detaches_pinned_commit(self, run, validate_checkout):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp) / "official"
            ensure_checkout(checkout)
            self.assertEqual(
                run.call_args_list,
                [
                    mock.call(["git", "clone", PROTOCOL.upstream_repo, str(checkout)], check=True),
                    mock.call(
                        ["git", "checkout", "--detach", PROTOCOL.upstream_commit],
                        cwd=checkout,
                        check=True,
                    ),
                ],
            )
            validate_checkout.assert_called_once_with(checkout)

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.validate_checkout")
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_ensure_existing_checkout_never_runs_mutating_git(self, run, validate_checkout):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            ensure_checkout(checkout)
            run.assert_not_called()
            validate_checkout.assert_called_once_with(checkout)

    @mock.patch(
        "lexbrowser_eval.lexbench.qwen3_8b.official.shutil.which", return_value="/usr/bin/tool"
    )
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.platform.system", return_value="Linux")
    def test_validate_environment_rejects_model_identity_mismatch(self, _system, _which):
        cases = (
            ("QWEN_MODEL_ID", "wrong-model", "QWEN_MODEL_ID=qwen3_8B"),
            ("LEXBENCH_JUDGE_MODEL", "wrong-judge", "LEXBENCH_JUDGE_MODEL=gpt-5.4"),
        )
        for name, value, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                environ = {item: "configured" for item in required_env_names(("lexmount",))}
                environ["QWEN_MODEL_ID"] = "qwen3_8B"
                environ["LEXBENCH_JUDGE_MODEL"] = "gpt-5.4"
                environ[name] = value
                with self.assertRaisesRegex(RuntimeError, expected):
                    validate_environment(
                        pathlib.Path("/data/wf/sxh"),
                        pathlib.Path(tmp),
                        ("lexmount",),
                        environ,
                    )

    @mock.patch(
        "lexbrowser_eval.lexbench.qwen3_8b.official.shutil.which", return_value="/usr/bin/tool"
    )
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.platform.system", return_value="Linux")
    def test_validate_environment_report_and_errors_do_not_leak_values(self, _system, _which):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            environ = {
                name: f"secret-sentinel-{index}"
                for index, name in enumerate(required_env_names(("lexmount",)))
            }
            environ["QWEN_MODEL_ID"] = "qwen3_8B"
            environ["LEXBENCH_JUDGE_MODEL"] = "gpt-5.4"
            report = validate_environment(
                pathlib.Path("/data/wf/sxh"), checkout / "results", ("lexmount",), environ
            )
            serialized = json.dumps(report, sort_keys=True)
            for value in environ.values():
                self.assertNotIn(value, serialized)
            missing = dict(environ)
            missing["QWEN_API_KEY"] = ""
            with self.assertRaisesRegex(RuntimeError, "QWEN_API_KEY") as raised:
                validate_environment(
                    pathlib.Path("/data/wf/sxh"), checkout / "results", ("lexmount",), missing
                )
            for value in environ.values():
                self.assertNotIn(value, str(raised.exception))

    def test_freeze_dependencies_runs_real_normalizing_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            root_python = checkout / ".venv" / "bin" / "python"
            browser_python = checkout / ".venvs" / "browser_use" / "bin" / "python"
            root_python.parent.mkdir(parents=True)
            browser_python.parent.mkdir(parents=True)
            root_python.symlink_to(sys.executable)
            browser_python.symlink_to(sys.executable)
            output = checkout / "dependencies.txt"
            freeze_dependencies(checkout, output)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines.count("[official-root]"), 1)
            self.assertEqual(lines.count("[browser-use]"), 1)
            for line in lines:
                if not line or line.startswith("["):
                    continue
                self.assertRegex(line, r"^[a-z0-9]+(?:-[a-z0-9]+)*==[A-Za-z0-9][A-Za-z0-9.!+_-]*$")
                self.assertNotIn("@", line)
                self.assertNotIn("://", line)

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_freeze_dependencies_records_names_and_versions_only(self, run):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            root_python = checkout / ".venv" / "bin" / "python"
            browser_python = checkout / ".venvs" / "browser_use" / "bin" / "python"
            root_python.parent.mkdir(parents=True)
            browser_python.parent.mkdir(parents=True)
            root_python.touch()
            browser_python.touch()
            run.side_effect = [
                subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps([["Zeta_Pkg", "2.0"], ["alpha", "01.0"]])
                ),
                subprocess.CompletedProcess([], 0, stdout=json.dumps([["Browser.Use", "3.0rc1"]])),
            ]
            output = checkout / "dependencies.txt"
            freeze_dependencies(checkout, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "[official-root]\nalpha==1.0\nzeta-pkg==2.0\n\n"
                "[browser-use]\nbrowser-use==3.0rc1\n",
            )
            for call in run.call_args_list:
                self.assertEqual(call.args[0][1], "-c")
                script = call.args[0][2]
                self.assertIn("importlib.metadata", script)
                self.assertNotIn("pip", script.lower())

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_freeze_requires_both_environments_without_touching_output(self, run):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            root_python = checkout / ".venv" / "bin" / "python"
            root_python.parent.mkdir(parents=True)
            root_python.touch()
            run.return_value = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps([["alpha", "1.0"]])
            )
            output = checkout / "dependencies.txt"
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "browser-use"):
                freeze_dependencies(checkout, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            run.assert_not_called()

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_freeze_rejects_invalid_payload_without_touching_output(self, run):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            for python in (
                checkout / ".venv" / "bin" / "python",
                checkout / ".venvs" / "browser_use" / "bin" / "python",
            ):
                python.parent.mkdir(parents=True)
                python.touch()
            sentinel = "json-traceback-secret-sentinel"
            run.return_value = subprocess.CompletedProcess([], 0, stdout=sentinel)
            output = checkout / "dependencies.txt"
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "Invalid dependency metadata payload"
            ) as raised:
                freeze_dependencies(checkout, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            exception = raised.exception
            formatted = "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
            self.assertNotIn(sentinel, formatted)
            self.assertIsNone(exception.__cause__)

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_freeze_rejects_malformed_records_without_leaking_or_touching_output(self, run):
        name_sentinel = "name-traceback-secret sentinel @"
        version_sentinel = "version-traceback-secret://value"
        cases = (
            (json.dumps({"alpha": "1.0"}), "Invalid dependency metadata payload", None),
            (json.dumps([["alpha"]]), "Invalid dependency metadata record", None),
            (json.dumps([]), "No dependency metadata records", None),
            (
                json.dumps([[name_sentinel, "1.0"]]),
                "Invalid dependency name metadata",
                name_sentinel,
            ),
            (
                json.dumps([["alpha", version_sentinel]]),
                "Invalid dependency version metadata",
                version_sentinel,
            ),
            (
                json.dumps([["Alpha", "1.0"], ["alpha", "2.0"]]),
                "Conflicting installed versions for alpha",
                None,
            ),
        )
        for payload, expected, sentinel in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                checkout = pathlib.Path(tmp)
                for python in (
                    checkout / ".venv" / "bin" / "python",
                    checkout / ".venvs" / "browser_use" / "bin" / "python",
                ):
                    python.parent.mkdir(parents=True)
                    python.touch()
                run.reset_mock()
                run.return_value = subprocess.CompletedProcess([], 0, stdout=payload)
                output = checkout / "dependencies.txt"
                output.write_text("sentinel\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, expected) as raised:
                    freeze_dependencies(checkout, output)
                self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
                self.assertNotIn(payload, str(raised.exception))
                self.assertNotIn("secret", str(raised.exception))
                if sentinel is not None:
                    exception = raised.exception
                    formatted = "".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    )
                    self.assertNotIn(sentinel, formatted)
                    self.assertIsNone(exception.__cause__)

    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_freeze_second_environment_failure_preserves_output(self, run):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            for python in (
                checkout / ".venv" / "bin" / "python",
                checkout / ".venvs" / "browser_use" / "bin" / "python",
            ):
                python.parent.mkdir(parents=True)
                python.touch()
            run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout=json.dumps([["alpha", "1.0"]])),
                subprocess.CalledProcessError(1, ["browser-python", "-c"]),
            ]
            output = checkout / "dependencies.txt"
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                freeze_dependencies(checkout, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(run.call_count, 2)

    @mock.patch(
        "lexbrowser_eval.lexbench.qwen3_8b.official.os.replace",
        side_effect=OSError("replace failed"),
    )
    @mock.patch("lexbrowser_eval.lexbench.qwen3_8b.official.subprocess.run")
    def test_freeze_replace_failure_preserves_output_and_cleans_temp(self, run, _replace):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            for python in (
                checkout / ".venv" / "bin" / "python",
                checkout / ".venvs" / "browser_use" / "bin" / "python",
            ):
                python.parent.mkdir(parents=True)
                python.touch()
            run.return_value = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps([["alpha", "1.0"]])
            )
            output = checkout / "dependencies.txt"
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(OSError, "replace failed"):
                freeze_dependencies(checkout, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(list(checkout.glob(".dependencies.txt.*")), [])


if __name__ == "__main__":
    unittest.main()
