from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "training" / "nvidia"


class DeliveryHarnessTests(unittest.TestCase):
    def test_dry_run_creates_secret_free_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    "bash",
                    str(DELIVERY / "run_nvidia.sh"),
                    "--mode",
                    "dry-run",
                    "--nodes",
                    "1",
                    "--gpus-per-node",
                    "1",
                    "--run-root",
                    temporary,
                    "--run-id",
                    "test-dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("dry-run manifest", completed.stdout)
            manifest = json.loads(
                (Path(temporary) / "test-dry-run" / "manifests" / "run_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["topology"]["gpu_family"], "H100")
            self.assertFalse(manifest["secrets"]["file_present"])
            self.assertFalse(any(manifest["secrets"]["required_names_present"].values()))

    def test_resume_path_requires_safe_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rejected = subprocess.run(
                [
                    "bash",
                    str(DELIVERY / "run_nvidia.sh"),
                    "--mode",
                    "dry-run",
                    "--run-root",
                    temporary,
                    "--run-id",
                    "unsafe-resume",
                    "--resume",
                    "/checkpoints/it's-here",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("resume path must be an absolute path", rejected.stderr)

            accepted = subprocess.run(
                [
                    "bash",
                    str(DELIVERY / "run_nvidia.sh"),
                    "--mode",
                    "dry-run",
                    "--run-root",
                    temporary,
                    "--run-id",
                    "safe-resume",
                    "--resume",
                    "/shared/checkpoints/run-1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("dry-run manifest", accepted.stdout)

    def test_cdp_preflight_uses_word_boundary_error_regex(self) -> None:
        source = (ROOT / "training" / "scripts" / "smoke_lexmount_cdp.py").read_text()
        self.assertIn('re.search(r"\\bERR_[A-Z_]+\\b", text)', source)
        self.assertNotIn('re.search(r"\\\\bERR_[A-Z_]+\\\\b", text)', source)

    def test_delivery_defaults_to_dmx_gpt_5_5(self) -> None:
        secrets_example = (DELIVERY / "secrets.env.example").read_text()
        self.assertIn("OPENAI_BASE_URL=https://www.dmxapi.cn/v1", secrets_example)
        self.assertIn("OPENAI_MODEL=gpt-5.5", secrets_example)
        for config in (
            ROOT / "training" / "nemo_gym" / "lexbrowser_webvoyager.yaml",
            ROOT / "training" / "lexbrowser_webvoyager.yaml",
        ):
            source = config.read_text()
            self.assertIn("judge_model: gpt-5.5", source)
            self.assertIn("stagehand_model: openai/gpt-5.5", source)
        for environment in (
            ROOT / "training" / "nemo_gym" / "environment.py",
            ROOT
            / "training"
            / "lexbrowser_webvoyager"
            / "src"
            / "lexbrowser_webvoyager_no_anti_bot"
            / "environment.py",
        ):
            self.assertIn('os.environ.get("OPENAI_MODEL")', environment.read_text())

    def test_preflight_validator_accepts_distinct_gpu_family_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nodes = root / "nodes"
            nodes.mkdir()
            for index in range(2):
                payload = {
                    "hostname": f"h{index}",
                    "checks": {
                        "gpu_count": True,
                        "gpu_family": True,
                        "gpu_memory": True,
                        "shared_storage_writable": True,
                    },
                    "gpus": [{"uuid": f"GPU-{index}-a"}, {"uuid": f"GPU-{index}-b"}],
                }
                (nodes / f"h{index}.json").write_text(json.dumps(payload))
            subprocess.run(
                [
                    "python3",
                    str(DELIVERY / "scripts" / "validate_preflight.py"),
                    "--nodes-dir",
                    str(nodes),
                    "--output",
                    str(root / "summary.json"),
                    "--expected-nodes",
                    "2",
                    "--expected-gpus-per-node",
                    "2",
                    "--expected-gpu-family",
                    "H100",
                ],
                check=True,
            )
            self.assertTrue(json.loads((root / "summary.json").read_text())["passed"])

    def test_backend_comparison_requires_matched_contract(self) -> None:
        contract = json.loads((DELIVERY / "configs" / "comparison_contract.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backend in ("lexmount", "local"):
                run = root / backend
                (run / "manifests").mkdir(parents=True)
                (run / "metrics").mkdir()
                manifest = {
                    "backend": backend,
                    "comparison_contract": contract,
                    "topology": {"nodes": 8, "gpus_per_node": 8},
                    "model": {"id": "Qwen/Qwen3-1.7B", "resolved_revision": "pinned"},
                    "source": {"dataset_sha256": contract["dataset_source_sha256"]},
                }
                (run / "manifests" / "run_manifest.json").write_text(json.dumps(manifest))
                (run / "metrics" / "resources_summary.json").write_text(
                    json.dumps({"training": {"last_avg_reward": 0.5, "last_loss": 1.0}})
                )
            output = root / "comparison.json"
            subprocess.run(
                [
                    "python3",
                    str(DELIVERY / "scripts" / "compare_backends.py"),
                    "--lexmount-run",
                    str(root / "lexmount"),
                    "--local-run",
                    str(root / "local"),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            self.assertTrue(json.loads(output.read_text())["comparable"])

            local_manifest = root / "local" / "manifests" / "run_manifest.json"
            changed = json.loads(local_manifest.read_text())
            changed["model"]["resolved_revision"] = "different-revision"
            local_manifest.write_text(json.dumps(changed))
            mismatch = subprocess.run(
                [
                    "python3",
                    str(DELIVERY / "scripts" / "compare_backends.py"),
                    "--lexmount-run",
                    str(root / "lexmount"),
                    "--local-run",
                    str(root / "local"),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(mismatch.returncode, 2)
            self.assertFalse(json.loads(output.read_text())["comparable"])

    def test_shell_scripts_parse(self) -> None:
        for script in sorted(DELIVERY.rglob("*.sh")):
            subprocess.run(["bash", "-n", str(script)], check=True)


if __name__ == "__main__":
    unittest.main()
