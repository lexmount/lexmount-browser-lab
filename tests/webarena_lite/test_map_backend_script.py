import os
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "setup_webarena_map_backend.sh"
RUNNER = ROOT / "run_webarena_lite.py"


class MapBackendScriptTests(unittest.TestCase):
    def test_root_python_entrypoint_exists(self):
        self.assertTrue(RUNNER.is_file())
        self.assertIn(
            "lexbrowser_eval.webarena_lite.cli",
            RUNNER.read_text(encoding="utf-8"),
        )

    def test_print_plan_keeps_every_persistent_path_on_data(self):
        env = dict(os.environ)
        env["WEBARENA_ROOT"] = "/data/wf/sxh"

        proc = subprocess.run(
            ["bash", str(SCRIPT), "--print-plan"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("root=/data/wf/sxh", proc.stdout)
        self.assertIn("downloads=/data/wf/sxh/webarena_map_backend/downloads", proc.stdout)
        self.assertIn("docker_socket=/data/wf/sxh/webarena_docker/docker.sock", proc.stdout)
        self.assertIn("osrm_ports=15000,15001,15002", proc.stdout)
        self.assertIn("container_memory_limit=30g", proc.stdout)
        self.assertIn("tile_url=http://10.2.131.41:8080/tile/10/284/385.png", proc.stdout)
        self.assertNotIn("downloads=/root", proc.stdout)
        self.assertNotIn("downloads=/opt", proc.stdout)

    def test_refuses_runtime_root_outside_data_mount(self):
        env = dict(os.environ)
        env["WEBARENA_ROOT"] = "/home/wf/sxh"

        proc = subprocess.run(
            ["bash", str(SCRIPT), "--print-plan"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("must be under /data", proc.stdout)

    def test_uses_archive_specific_volume_prefix_depths(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            "extract_volume_archive osm_tile_server.tar tile.extracted 4",
            source,
        )
        self.assertIn(
            "extract_volume_archive nominatim_volumes.tar nominatim.extracted 5",
            source,
        )

    def test_backend_mounts_and_limits_match_preloaded_data(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('OSRM_MEMORY_LIMIT="8g"', source)
        self.assertIn(
            '${MAP_ROOT}/osm_dump/osm_dump:/nominatim/data"',
            source,
        )
        self.assertIn("pg_ctl_options = '-t 600'", source)
        self.assertNotIn('${MAP_ROOT}/osm_dump:/nominatim/data:ro', source)


if __name__ == "__main__":
    unittest.main()
