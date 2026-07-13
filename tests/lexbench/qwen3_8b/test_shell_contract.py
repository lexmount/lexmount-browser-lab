import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[3]


class CampaignScriptTests(unittest.TestCase):
    def test_unified_shell_delegates_qwen_to_package_interface(self):
        script = (ROOT / "scripts" / "run_lexbench.sh").read_text(encoding="utf-8")
        self.assertIn("lexbrowser_eval.lexbench.cli", script)
        self.assertIn("qwen3-8b", script)

    def test_unified_shell_preserves_gpt55_adapter(self):
        script = (ROOT / "scripts" / "run_lexbench.sh").read_text(encoding="utf-8")
        self.assertIn("run_gpt55_lexbench.sh", script)

    def test_gpt55_adapter_invokes_src_modules(self):
        script = (ROOT / "scripts" / "run_gpt55_lexbench.sh").read_text(encoding="utf-8")
        self.assertIn("lexbrowser_eval.resources", script)
        self.assertIn("lexbrowser_eval.lexbench.summarize", script)
        self.assertNotIn("scripts/profile_", script)


if __name__ == "__main__":
    unittest.main()
