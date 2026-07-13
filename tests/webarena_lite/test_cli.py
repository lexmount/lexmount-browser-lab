import datetime as dt
import io
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from lexbrowser_eval.webarena_lite import cli as runner


class RunWebArenaLiteTests(unittest.TestCase):
    def setUp(self):
        self.runner = runner

    def test_console_entrypoint_accepts_no_explicit_argv(self):
        with mock.patch.object(sys, "argv", ["webarena-lite-eval", "--help"]):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.runner.main()

        self.assertEqual(raised.exception.code, 0)

    def test_default_result_dir_uses_model_slug_and_timestamp(self):
        now = dt.datetime(2026, 7, 9, 12, 34, 56)

        result = self.runner.default_result_dir(pathlib.Path("/repo"), "glm-5.2", now=now)

        self.assertEqual(
            result,
            pathlib.Path("/repo/results/glm52_webarena_lite_20260709_123456"),
        )

    def test_default_runtime_root_prefers_data_mount_when_available(self):
        def exists(path):
            return path == pathlib.Path("/data/wf/sxh")

        def writable(path):
            return path == pathlib.Path("/data/wf/sxh")

        root = self.runner.default_runtime_root(
            pathlib.Path("/home/wf/sxh"),
            exists=exists,
            writable=writable,
        )

        self.assertEqual(root, pathlib.Path("/data/wf/sxh"))

    def test_harness_env_maps_openai_base_url_and_site_defaults(self):
        environ = {
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://litellm.local.lexmount.net/v1",
            "OPENAI_MODEL": "glm-5.2",
        }

        env = self.runner.build_harness_env(
            environ,
            server="webarena.internal",
            map_server=None,
        )

        self.assertEqual(env["OPENAI_API_URL"], environ["OPENAI_BASE_URL"])
        self.assertEqual(env["DATASET"], "webarena")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["SHOPPING"], "http://webarena.internal:7770")
        self.assertEqual(env["SHOPPING_ADMIN"], "http://webarena.internal:7780/admin")
        self.assertEqual(env["REDDIT"], "http://webarena.internal:9999")
        self.assertEqual(env["GITLAB"], "http://webarena.internal:8023")
        self.assertEqual(env["MAP"], "http://webarena.internal:3000")
        self.assertEqual(env["HOMEPAGE"], "http://webarena.internal:4399")
        self.assertEqual(
            env["WIKIPEDIA"],
            "http://webarena.internal:8888/"
            "wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        )

    def test_existing_site_environment_overrides_defaults(self):
        environ = {
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://litellm.local.lexmount.net/v1",
            "OPENAI_MODEL": "glm-5.2",
            "SHOPPING": "http://custom-shopping:7770",
        }

        env = self.runner.build_harness_env(
            environ,
            server="webarena.internal",
            map_server="map.internal",
        )

        self.assertEqual(env["SHOPPING"], "http://custom-shopping:7770")
        self.assertEqual(env["MAP"], "http://map.internal:3000")

    def test_load_site_env_uses_file_without_overriding_existing_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            env_dir = root / "webarena_env"
            env_dir.mkdir()
            (env_dir / "site_env.sh").write_text(
                "SHOPPING=http://from-file:7770\nMAP=http://from-file:13000\nIGNORED=value\n",
                encoding="utf-8",
            )

            env = self.runner.environ_with_site_env(
                root,
                {
                    "SHOPPING": "http://explicit:7770",
                    "OPENAI_API_KEY": "secret",
                },
            )

        self.assertEqual(env["SHOPPING"], "http://explicit:7770")
        self.assertEqual(env["MAP"], "http://from-file:13000")
        self.assertNotIn("IGNORED", env)

    def test_run_command_matches_webrl_lite_defaults(self):
        config = self.runner.RunConfig(
            result_dir=pathlib.Path("/repo/results/run1"),
            test_start_idx=0,
            test_end_idx=165,
            model="glm-5.2",
        )

        command = self.runner.build_run_command("/venv/bin/python", config)

        self.assertEqual(command[:2], ["/venv/bin/python", "run.py"])
        self.assertIn("--instruction_path", command)
        self.assertIn("agent/prompts/jsons/p_webrl_chat.json", command)
        self.assertIn("--test_config_base_dir", command)
        self.assertIn("config_files/wa/test_webarena_lite", command)
        self.assertIn("--provider", command)
        self.assertIn("openai", command)
        self.assertIn("--model", command)
        self.assertIn("glm-5.2", command)
        self.assertIn("--action_set_tag", command)
        self.assertIn("webrl_id", command)
        self.assertIn("--observation_type", command)
        self.assertIn("webrl", command)
        self.assertIn("--max_steps", command)
        self.assertIn("30", command)

    def test_env_with_venv_python_prefers_virtualenv_bin(self):
        env = {"PATH": "/usr/bin:/bin"}

        updated = self.runner.env_with_venv_python(
            env, pathlib.Path("/repo/.venv-walite/bin/python")
        )

        self.assertEqual(updated["PATH"], "/repo/.venv-walite/bin:/usr/bin:/bin")
        self.assertEqual(updated["NLTK_DATA"], "/repo/.venv-walite/nltk_data")

    def test_runtime_dependencies_pin_missing_upstream_packages(self):
        self.assertEqual(
            self.runner.RUNTIME_REQUIREMENTS,
            (
                "lxml==4.9.3",
                "dashscope==1.14.1",
                "anthropic==0.4.1",
            ),
        )

    def test_env_with_harness_pythonpath_prefers_harness_root(self):
        env = {"PYTHONPATH": "/existing"}

        updated = self.runner.env_with_harness_pythonpath(
            env, pathlib.Path("/repo/VAB-WebArena-Lite")
        )

        self.assertEqual(updated["PYTHONPATH"], "/repo/VAB-WebArena-Lite:/existing")

    def test_patch_webrl_action_parser_turns_missing_action_into_parse_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness_dir = pathlib.Path(tmp)
            actions_path = harness_dir / "browser_env" / "actions.py"
            actions_path.parent.mkdir()
            actions_path.write_text(
                '        case "do":\n'
                '            action_type = action["action"].lower()\n'
                "            match action_type:\n",
                encoding="utf-8",
            )

            self.runner.patch_webrl_action_parser(harness_dir)
            first_patch = actions_path.read_text(encoding="utf-8")
            self.runner.patch_webrl_action_parser(harness_dir)

            self.assertIn(
                'action_type = str(action.get("action", "")).lower()',
                first_patch,
            )
            self.assertEqual(
                actions_path.read_text(encoding="utf-8"),
                first_patch,
            )


if __name__ == "__main__":
    unittest.main()
