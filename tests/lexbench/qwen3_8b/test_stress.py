from __future__ import annotations

import copy
import hashlib
import json
import unittest

from lexbrowser_eval.lexbench.qwen3_8b.stress import (
    BASE_TASK_COUNT,
    EXPECTED_DATASET_SHA256,
    EXPECTED_TASK_IDS,
    EXPECTED_TASK_IDS_SHA256,
    RANDOM_SEED,
    TARGETS,
    build_official_replica_command,
    build_stress_manifest,
    canonical_manifest_sha256,
    replica_count,
    select_task_ids_from_jsonl,
    task_ids_sha256,
    validate_config_snapshot,
    validate_frozen_sample,
)


def _jsonl(ids: list[str]) -> bytes:
    return b"".join(
        json.dumps({"id": int(task_id), "query": f"task {task_id}"}).encode() + b"\n"
        for task_id in ids
    )


class FrozenSampleTests(unittest.TestCase):
    def test_frozen_protocol_constants(self):
        self.assertEqual(RANDOM_SEED, 20260710)
        self.assertEqual(BASE_TASK_COUNT, 20)
        self.assertEqual(TARGETS, (20, 60, 100, 200, 500))
        self.assertEqual(
            EXPECTED_TASK_IDS,
            (
                "42",
                "48",
                "68",
                "74",
                "98",
                "137",
                "141",
                "151",
                "168",
                "172",
                "216",
                "236",
                "241",
                "302",
                "306",
                "3004",
                "3009",
                "3011",
                "2008",
                "2011",
            ),
        )
        self.assertEqual(
            EXPECTED_DATASET_SHA256,
            "90fc09b2fbdcd391d70924d1fee069534784bc133aaefabf26d3892e48983108",
        )
        self.assertEqual(
            EXPECTED_TASK_IDS_SHA256,
            "16cf59aa09cdd05e08401e5c62ca3ed4f9a7c737f6d4463054a97abc12d5519d",
        )
        self.assertEqual(task_ids_sha256(EXPECTED_TASK_IDS), EXPECTED_TASK_IDS_SHA256)

    def test_selection_is_seeded_but_returned_in_dataset_order(self):
        payload = _jsonl([str(number) for number in range(1, 31)])
        first = select_task_ids_from_jsonl(payload, seed=17, sample_size=5)
        second = select_task_ids_from_jsonl(payload, seed=17, sample_size=5)
        self.assertEqual(first, second)
        self.assertEqual(first, tuple(sorted(first, key=int)))
        self.assertEqual(len(first), 5)

    def test_validate_frozen_sample_checks_dataset_ids_and_ids_hash(self):
        payload = _jsonl([str(number) for number in range(1, 31)])
        selected = select_task_ids_from_jsonl(payload, seed=17, sample_size=5)
        validated = validate_frozen_sample(
            payload,
            seed=17,
            sample_size=5,
            expected_dataset_sha256=hashlib.sha256(payload).hexdigest(),
            expected_task_ids=selected,
            expected_task_ids_sha256=task_ids_sha256(selected),
        )
        self.assertEqual(validated, selected)

        with self.assertRaisesRegex(ValueError, "Dataset SHA-256 mismatch"):
            validate_frozen_sample(
                payload,
                seed=17,
                sample_size=5,
                expected_dataset_sha256="0" * 64,
                expected_task_ids=selected,
                expected_task_ids_sha256=task_ids_sha256(selected),
            )


class ReplicaCommandTests(unittest.TestCase):
    def test_replica_count_for_each_target(self):
        self.assertEqual([replica_count(target) for target in TARGETS], [1, 3, 5, 10, 25])
        with self.assertRaisesRegex(ValueError, "frozen target"):
            replica_count(40)

    def test_lexmount_command_uses_one_isolated_official_run(self):
        command = build_official_replica_command("lexmount", "20260710_230001", ("42", "48"))
        self.assertEqual(
            command,
            [
                "uv",
                "run",
                "bubench",
                "run",
                "--agent",
                "browser-use",
                "--data",
                "LexBench-Browser",
                "--model",
                "qwen3-8B",
                "--browser-id",
                "lexmount",
                "--split",
                "All",
                "--mode",
                "specific",
                "--task-ids",
                "42",
                "48",
                "--no-group-by-site",
                "--concurrency",
                "20",
                "--timestamp",
                "20260710_230001",
            ],
        )

    def test_local_command_wraps_the_same_official_command_in_xvfb(self):
        command = build_official_replica_command("local", "20260710_230002", ("42",))
        self.assertEqual(
            command[:6],
            ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", "uv", "run"],
        )
        self.assertEqual(command[6:9], ["bubench", "run", "--agent"])
        self.assertEqual(command[command.index("--browser-id") + 1], "local")
        self.assertEqual(command[command.index("--timestamp") + 1], "20260710_230002")

    def test_command_rejects_invalid_backend_timestamp_or_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            build_official_replica_command("other", "20260710_230003", ("42",))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            build_official_replica_command("local", "bad", ("42",))
        with self.assertRaisesRegex(ValueError, "unique"):
            build_official_replica_command("local", "20260710_230003", ("42", "42"))


class ManifestAndSnapshotTests(unittest.TestCase):
    def test_manifest_binds_sample_backend_target_and_unique_timestamps(self):
        timestamps = tuple(f"20260710_23{minute:02d}00" for minute in range(3))
        manifest = build_stress_manifest("lexmount", 60, timestamps, EXPECTED_TASK_IDS)
        self.assertEqual(manifest["seed"], RANDOM_SEED)
        self.assertEqual(manifest["target_concurrency"], 60)
        self.assertEqual(manifest["replica_count"], 3)
        self.assertEqual(manifest["per_replica_concurrency"], 20)
        self.assertEqual(manifest["task_ids_sha256"], EXPECTED_TASK_IDS_SHA256)
        self.assertEqual([item["timestamp"] for item in manifest["replicas"]], list(timestamps))

        with self.assertRaisesRegex(ValueError, "timestamp count"):
            build_stress_manifest("lexmount", 60, timestamps[:2], EXPECTED_TASK_IDS)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_stress_manifest("lexmount", 60, (timestamps[0],) * 3, EXPECTED_TASK_IDS)

    def test_manifest_hash_is_canonical_and_detects_changes(self):
        manifest = {"target": 60, "tasks": ["42", "48"], "backend": "local"}
        reordered = {"backend": "local", "tasks": ["42", "48"], "target": 60}
        self.assertEqual(canonical_manifest_sha256(manifest), canonical_manifest_sha256(reordered))
        changed = {**manifest, "target": 100}
        self.assertNotEqual(canonical_manifest_sha256(manifest), canonical_manifest_sha256(changed))

    def test_config_snapshot_validator_checks_frozen_runtime_contract(self):
        snapshot = {
            "run": {
                "agent": "browser-use",
                "benchmark": "LexBench-Browser",
                "split": "All",
                "model_id": "qwen3_8B",
                "timestamp": "20260710_230001",
                "model_name_override": "qwen3-8B",
                "browser_id_override": "local",
                "mode": "specific",
                "task_ids": list(EXPECTED_TASK_IDS),
                "concurrency": 20,
            },
            "runtime_config": {
                "browser_id": "local",
                "model_id": "qwen3_8B",
                "max_steps": 40,
                "timeout": 600,
                "flash_mode": True,
                "use_vision": False,
                "use_judge": False,
                "dont_force_structured_output": False,
                "add_schema_to_system_prompt": True,
            },
        }
        validate_config_snapshot(
            snapshot, backend="local", timestamp="20260710_230001", task_ids=EXPECTED_TASK_IDS
        )

        broken = copy.deepcopy(snapshot)
        broken["runtime_config"]["max_steps"] = 39
        with self.assertRaisesRegex(ValueError, "runtime_config.max_steps"):
            validate_config_snapshot(
                broken,
                backend="local",
                timestamp="20260710_230001",
                task_ids=EXPECTED_TASK_IDS,
            )


if __name__ == "__main__":
    unittest.main()
