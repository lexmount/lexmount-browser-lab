from __future__ import annotations

import json
import pathlib
import tempfile
import threading
import time
import unittest

from lexbrowser_eval.lexbench.qwen3_8b import stress_stage2 as MODULE


def _write_result(task_dir: pathlib.Path, payload: object) -> pathlib.Path:
    task_dir.mkdir(parents=True)
    path = task_dir / "result.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class DiscoveryAndIsolationTests(unittest.TestCase):
    def test_discovers_only_valid_results_and_distinguishes_repeated_task_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run_a = root / "run-a"
            run_b = root / "run-b"
            _write_result(run_a / "tasks" / "42", {"task_id": "42", "metrics": {"steps": 3}})
            bad = run_a / "tasks" / "48"
            bad.mkdir(parents=True)
            (bad / "result.json").write_text("{bad", encoding="utf-8")
            _write_result(run_b / "tasks" / "42", {"task_id": "42", "metrics": {"steps": 4}})
            rollout = {
                "cells": {
                    "lexmount": [
                        {
                            "backend": "lexmount",
                            "target_concurrency": 40,
                            "cell_name": "lexmount_c40",
                            "official_run_dirs": [str(run_a), str(run_b)],
                        }
                    ]
                }
            }

            instances, excluded = MODULE.discover_instances(rollout)

        self.assertEqual(len(instances), 2)
        self.assertEqual(
            [item.replica_key for item in instances],
            [
                "lexmount/lexmount_c40/r00",
                "lexmount/lexmount_c40/r01",
            ],
        )
        self.assertEqual(
            [item.tasks[0].instance_key for item in instances],
            [
                "lexmount/lexmount_c40/r00/42",
                "lexmount/lexmount_c40/r01/42",
            ],
        )
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["task_id"], "48")
        self.assertIn("invalid_json", excluded[0]["reason"])

    def test_prepare_creates_only_stage_symlinks_and_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run_dir = root / "official-run"
            source = _write_result(
                run_dir / "tasks" / "42",
                {"task_id": "42", "metrics": {"steps": 3, "end_to_end_ms": 1200}},
            )
            rollout = {
                "cells": {
                    "local": [
                        {
                            "backend": "local",
                            "target_concurrency": 20,
                            "cell_name": "local_c20",
                            "official_run_dirs": [str(run_dir)],
                        }
                    ]
                }
            }
            stage = root / "campaign" / "stage2_gpt54_c5"

            manifest = MODULE.prepare_stage(
                checkout=root / "checkout",
                rollout_summary=rollout,
                rollout_summary_path=root / "rollout_summary.json",
                stage_dir=stage,
                workers=5,
                official_commit="test-commit",
            )

            link = stage / "instances" / "local__local_c20__r00" / "tasks" / "42"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (run_dir / "tasks" / "42").resolve())
            self.assertFalse((run_dir / "tasks_eval_result").exists())
            record = manifest["instances"][0]["tasks"][0]
            self.assertEqual(record["sha256"], MODULE.sha256_file(source))
            self.assertEqual(record["mtime_ns"], source.stat().st_mtime_ns)
            MODULE.validate_sources_unchanged(manifest)

            source.write_text('{"task_id":"42","changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Source result changed"):
                MODULE.validate_sources_unchanged(manifest)


class ConcurrencyAndAggregationTests(unittest.TestCase):
    def test_bounded_runner_never_exceeds_global_limit(self):
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(item: int) -> int:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return item * 2

        results = MODULE.run_bounded(list(range(12)), max_workers=5, worker=worker)

        self.assertEqual(results, [value * 2 for value in range(12)])
        self.assertLessEqual(maximum, 5)
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            MODULE.run_bounded([1], max_workers=6, worker=worker)

    def test_enrichment_keeps_duplicate_task_ids_as_unique_instances(self):
        instance_a = MODULE.ReplicaInstance(
            backend="lexmount",
            cell="lexmount_c20",
            target_concurrency=20,
            replica_index=0,
            run_dir=pathlib.Path("/run/a"),
            stage_name="a",
            tasks=(
                MODULE.SourceTask(
                    "42",
                    pathlib.Path("/run/a/tasks/42"),
                    pathlib.Path("/run/a/tasks/42/result.json"),
                    "aa",
                    1,
                    {"metrics": {"steps": 3, "end_to_end_ms": 1000}},
                ),
            ),
        )
        instance_b = MODULE.ReplicaInstance(
            backend="local",
            cell="local_c20",
            target_concurrency=20,
            replica_index=0,
            run_dir=pathlib.Path("/run/b"),
            stage_name="b",
            tasks=(
                MODULE.SourceTask(
                    "42",
                    pathlib.Path("/run/b/tasks/42"),
                    pathlib.Path("/run/b/tasks/42/result.json"),
                    "bb",
                    2,
                    {"metrics": {"steps": 5, "end_to_end_ms": 3000}},
                ),
            ),
        )

        records = MODULE.enrich_instance_records(
            instance_a, [{"task_id": "42", "predicted_label": 1}]
        )
        records += MODULE.enrich_instance_records(
            instance_b, [{"task_id": "42", "predicted_label": 0}]
        )
        aggregate = MODULE.aggregate_records(records)

        self.assertEqual(len({record["stress_instance_key"] for record in records}), 2)
        self.assertEqual(aggregate["evaluated_instances"], 2)
        self.assertEqual(aggregate["successful_instances"], 1)
        self.assertEqual(aggregate["success_rate_percent"], 50.0)
        self.assertEqual(aggregate["avg_steps"], 4.0)
        self.assertEqual(aggregate["avg_e2e_seconds"], 2.0)

    def test_enrichment_rejects_missing_or_duplicate_official_task_records(self):
        instance = MODULE.ReplicaInstance(
            backend="local",
            cell="local_c20",
            target_concurrency=20,
            replica_index=0,
            run_dir=pathlib.Path("/run"),
            stage_name="r",
            tasks=(
                MODULE.SourceTask(
                    "42",
                    pathlib.Path("/run/tasks/42"),
                    pathlib.Path("/run/tasks/42/result.json"),
                    "aa",
                    1,
                    {},
                ),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            MODULE.enrich_instance_records(instance, [])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            MODULE.enrich_instance_records(
                instance,
                [{"task_id": "42"}, {"task_id": "42"}],
            )


if __name__ == "__main__":
    unittest.main()
