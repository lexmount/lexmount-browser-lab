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
        self.assertEqual(
            env["MAP_TILE"],
            "http://webarena.internal:3000/tile/10/284/385.png",
        )
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

    def test_explicit_webarena_server_ignores_stale_site_file_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            env_dir = root / "webarena_env"
            env_dir.mkdir()
            (env_dir / "site_env.sh").write_text(
                "SHOPPING=http://old-route-a:7770\n"
                "MAP=http://old-route-a:13000\n"
                "MAP_TILE=http://old-route-a:8080/tile/0/0/0.png\n",
                encoding="utf-8",
            )

            env = self.runner.environ_with_site_env(
                root,
                {
                    "WEBARENA_SERVER": "wa.example",
                    "OPENAI_API_KEY": "secret",
                },
            )

        self.assertNotIn("SHOPPING", env)
        self.assertNotIn("MAP", env)
        self.assertNotIn("MAP_TILE", env)

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
        self.assertEqual(updated["PLAYWRIGHT_BROWSERS_PATH"], "/repo/.playwright-browsers")
        self.assertEqual(updated["TMPDIR"], "/repo/.tmp")

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

    def test_patch_auto_login_timeout_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness_dir = pathlib.Path(tmp)
            auto_login_path = harness_dir / "browser_env" / "auto_login.py"
            auto_login_path.parent.mkdir()
            auto_login_path.write_text(
                "def first():\n"
                "    page = context.new_page()\n"
                "def second():\n"
                "    page = context.new_page()\n",
                encoding="utf-8",
            )

            self.runner.patch_auto_login_timeout(harness_dir)
            first_patch = auto_login_path.read_text(encoding="utf-8")
            self.runner.patch_auto_login_timeout(harness_dir)
            final_source = auto_login_path.read_text(encoding="utf-8")

        self.assertEqual(first_patch.count("page.set_default_timeout(120_000)"), 2)
        self.assertEqual(final_source, first_patch)

    def test_prepare_login_replaces_auth_only_after_all_states_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness_dir = pathlib.Path(tmp)
            auth_dir = harness_dir / ".auth"
            auth_dir.mkdir()
            (auth_dir / "old.json").write_text("old", encoding="utf-8")

            def fake_run(command, **kwargs):
                if "browser_env/auto_login.py" not in command:
                    return mock.Mock(returncode=0)
                auth_folder = pathlib.Path(command[command.index("--auth_folder") + 1])
                sites = command[command.index("--site_list") + 1 :]
                (auth_folder / f"{'.'.join(sites)}_state.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            with mock.patch.object(self.runner, "run", side_effect=fake_run) as run:
                self.runner.prepare_login(harness_dir, {}, skip=False)

            generated = {path.name for path in auth_dir.glob("*.json")}
            backups = list(harness_dir.glob(".auth.before-*"))
            backup_old = (backups[0] / "old.json").read_text(encoding="utf-8")

        self.assertEqual(
            generated,
            {
                "gitlab_state.json",
                "reddit_state.json",
                "shopping_admin_state.json",
                "shopping_state.json",
                "gitlab.reddit_state.json",
            },
        )
        self.assertEqual(len(backups), 1)
        self.assertEqual(backup_old, "old")
        self.assertEqual(run.call_count, 6)

    def test_runtime_probe_uses_harness_environment(self):
        python_bin = pathlib.Path("/runtime/bin/python")
        runtime_env = {"PLAYWRIGHT_BROWSERS_PATH": "/data/browsers"}
        completed = mock.Mock(returncode=0)

        with mock.patch.object(self.runner, "capture", return_value=completed) as capture:
            ready = self.runner.runtime_dependencies_ready(python_bin, runtime_env)

        self.assertTrue(ready)
        self.assertEqual(capture.call_args.kwargs["env"], runtime_env)
        self.assertFalse(capture.call_args.kwargs["check"])
        self.assertEqual(capture.call_args.args[0][0], str(python_bin))

    def test_http_404_is_unhealthy(self):
        error = self.runner.urllib.error.HTTPError(
            "http://map.internal/tile/0/0/0.png",
            404,
            "Not Found",
            {},
            None,
        )
        with mock.patch.object(self.runner.urllib.request, "urlopen", side_effect=error):
            ok, status = self.runner.http_status(
                "http://map.internal/tile/0/0/0.png", timeout=1.0
            )

        self.assertFalse(ok)
        self.assertEqual(status, "404")

    def test_site_health_checks_real_map_tile_endpoint(self):
        env = {
            "SHOPPING": "http://wa:7770",
            "SHOPPING_ADMIN": "http://wa:7780/admin",
            "REDDIT": "http://wa:9999",
            "GITLAB": "http://wa:8023",
            "MAP": "http://wa:13000",
            "MAP_TILE": "http://wa:8080/tile/10/284/385.png",
            "WIKIPEDIA": "http://wa:8888/wiki",
            "HOMEPAGE": "http://wa:4399",
        }
        with mock.patch.object(
            self.runner,
            "http_status",
            return_value=(True, "200"),
        ) as status:
            results = self.runner.check_sites(env)

        self.assertIn(
            ("MAP_TILE", "http://wa:8080/tile/10/284/385.png", True, "200"),
            results,
        )
        status.assert_any_call("http://wa:8080/tile/10/284/385.png", 5.0)

    def test_task_config_fingerprint_changes_when_a_site_url_changes(self):
        base = {
            "SHOPPING": "http://wa:7770",
            "SHOPPING_ADMIN": "http://wa:7780/admin",
            "REDDIT": "http://wa:9999",
            "GITLAB": "http://wa:8023",
            "MAP": "http://wa:13000",
            "WIKIPEDIA": "http://wa:8888/wiki",
            "HOMEPAGE": "http://wa:4399",
        }
        changed = dict(base, MAP="http://new-map:3000")

        self.assertNotEqual(
            self.runner.task_config_fingerprint(base),
            self.runner.task_config_fingerprint(changed),
        )

    def test_validate_results_rejects_negative_environment_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = pathlib.Path(tmp)
            actions = result_dir / "actions"
            actions.mkdir()
            (actions / "0.json").write_text(
                '{"task_id": 0, "score": -0.1, "actions": []}',
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                self.runner.validate_results(result_dir, [0])

    def test_smoke_report_uses_one_task_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            harness = root / "harness"
            result_dir = root / "results"
            config_dir = harness / "config_files" / "wa"
            actions = result_dir / "actions"
            config_dir.mkdir(parents=True)
            actions.mkdir(parents=True)
            (config_dir / "test_webarena_lite.raw.json").write_text(
                '[{"task_id": 0, "sites": ["shopping_admin"]}]',
                encoding="utf-8",
            )
            (actions / "0.json").write_text(
                '{"task_id": 0, "score": 0.0, "actions": ["click"]}',
                encoding="utf-8",
            )
            score_path = result_dir / "score.txt"
            score_path.write_text("successed: 0 / 1\n", encoding="utf-8")

            report_path = self.runner.write_report(
                harness,
                result_dir,
                "qwen3_8B",
                score_path,
                expected_task_ids=[0],
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("Finished tasks: 1/1", report)
        self.assertIn("Successful tasks: 0/1", report)
        self.assertIn("Overall SR: **0.00%**", report)
        self.assertNotIn("/165", report)


if __name__ == "__main__":
    unittest.main()
