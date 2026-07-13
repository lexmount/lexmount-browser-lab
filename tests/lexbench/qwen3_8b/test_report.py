import csv
import pathlib
import tempfile
import unittest

from lexbrowser_eval.lexbench.qwen3_8b.report import summarize_resource_files


class CampaignReportTests(unittest.TestCase):
    def test_resource_summary_uses_rollout_rows_and_valid_pss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            resource = root / "resource_samples.csv"
            gpu = root / "gpu_samples.csv"
            with resource.open("w", encoding="utf-8", newline="") as handle:
                fields = [
                    "phase",
                    "started_instance_count",
                    "cpu_cores",
                    "process_tree_pss_bytes",
                    "chrome_pss_bytes",
                    "pss_sample_age_seconds",
                    "memory_peak_bytes",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "phase": "baseline",
                            "started_instance_count": 0,
                            "cpu_cores": 99,
                            "process_tree_pss_bytes": 99,
                            "chrome_pss_bytes": 99,
                            "pss_sample_age_seconds": 0,
                            "memory_peak_bytes": 99,
                        },
                        {
                            "phase": "steady",
                            "started_instance_count": 20,
                            "cpu_cores": 2,
                            "process_tree_pss_bytes": 1073741824,
                            "chrome_pss_bytes": 536870912,
                            "pss_sample_age_seconds": 1,
                            "memory_peak_bytes": 2147483648,
                        },
                        {
                            "phase": "drain",
                            "started_instance_count": 20,
                            "cpu_cores": 4,
                            "process_tree_pss_bytes": 3221225472,
                            "chrome_pss_bytes": 1073741824,
                            "pss_sample_age_seconds": 2,
                            "memory_peak_bytes": 4294967296,
                        },
                        {
                            "phase": "drain",
                            "started_instance_count": 20,
                            "cpu_cores": 6,
                            "process_tree_pss_bytes": 999,
                            "chrome_pss_bytes": 999,
                            "pss_sample_age_seconds": 16,
                            "memory_peak_bytes": 3221225472,
                        },
                    ]
                )
            with gpu.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["phase", "gpu_sm_percent"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"phase": "baseline", "gpu_sm_percent": 0},
                        {"phase": "steady", "gpu_sm_percent": 80},
                        {"phase": "drain", "gpu_sm_percent": 100},
                    ]
                )

            summary = summarize_resource_files(resource, gpu)

        self.assertEqual(summary["cpu_cores_mean"], 4)
        self.assertEqual(summary["pss_gib_mean"], 2)
        self.assertEqual(summary["pss_gib_p95"], 3)
        self.assertEqual(summary["chrome_pss_gib_mean"], 0.75)
        self.assertEqual(summary["cgroup_memory_peak_gib"], 4)
        self.assertEqual(summary["gpu_sm_mean"], 90)
        self.assertEqual(summary["gpu_idle_mean"], 10)


if __name__ == "__main__":
    unittest.main()
