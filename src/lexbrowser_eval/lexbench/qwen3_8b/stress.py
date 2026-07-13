from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

RANDOM_SEED = 20260710
BASE_TASK_COUNT = 20
PER_REPLICA_CONCURRENCY = 20
TARGETS = (20, 60, 100, 200, 500)
BACKENDS = ("lexmount", "local")

EXPECTED_DATASET_SHA256 = "90fc09b2fbdcd391d70924d1fee069534784bc133aaefabf26d3892e48983108"
EXPECTED_TASK_IDS = (
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
)
EXPECTED_TASK_IDS_SHA256 = "16cf59aa09cdd05e08401e5c62ca3ed4f9a7c737f6d4463054a97abc12d5519d"

_TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def task_ids_sha256(task_ids: Sequence[str]) -> str:
    """Hash an ordered task-ID manifest with an unambiguous trailing newline."""
    normalized = tuple(str(task_id) for task_id in task_ids)
    payload = ("\n".join(normalized) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_ids_from_jsonl(payload: bytes) -> tuple[str, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Dataset is not valid UTF-8") from None

    task_ids: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSONL record at line {line_number}") from None
        if not isinstance(record, dict):
            raise ValueError(f"Non-object JSONL record at line {line_number}")
        raw_task_id = record.get("task_id") or record.get("annotation_id")
        if raw_task_id is None:
            raw_task_id = record.get("id")
        if raw_task_id is None or not str(raw_task_id).strip():
            raise ValueError(f"Missing task ID at line {line_number}")
        task_ids.append(str(raw_task_id))

    if not task_ids:
        raise ValueError("Dataset contains no tasks")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Dataset task IDs must be unique")
    return tuple(task_ids)


def select_task_ids_from_jsonl(
    payload: bytes,
    *,
    seed: int = RANDOM_SEED,
    sample_size: int = BASE_TASK_COUNT,
) -> tuple[str, ...]:
    """Select a reproducible sample and return it in official dataset order.

    The upstream ``specific`` mode treats the IDs as a set and filters the
    dataset in its original order. Returning that order makes the frozen
    manifest match the order the official runner actually executes.
    """
    task_ids = _task_ids_from_jsonl(payload)
    if sample_size < 1 or sample_size > len(task_ids):
        raise ValueError("sample_size must be between 1 and the dataset size")
    selected = set(random.Random(seed).sample(task_ids, sample_size))
    return tuple(task_id for task_id in task_ids if task_id in selected)


def validate_frozen_sample(
    payload: bytes,
    *,
    seed: int = RANDOM_SEED,
    sample_size: int = BASE_TASK_COUNT,
    expected_dataset_sha256: str = EXPECTED_DATASET_SHA256,
    expected_task_ids: Sequence[str] = EXPECTED_TASK_IDS,
    expected_task_ids_sha256: str = EXPECTED_TASK_IDS_SHA256,
) -> tuple[str, ...]:
    """Validate the official dataset bytes and the complete frozen sample."""
    actual_dataset_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "Dataset SHA-256 mismatch: "
            f"expected {expected_dataset_sha256}, got {actual_dataset_sha256}"
        )

    selected = select_task_ids_from_jsonl(payload, seed=seed, sample_size=sample_size)
    normalized_expected = tuple(str(task_id) for task_id in expected_task_ids)
    if selected != normalized_expected:
        raise ValueError(
            f"Frozen task IDs mismatch: expected {normalized_expected!r}, got {selected!r}"
        )
    actual_task_ids_sha256 = task_ids_sha256(selected)
    if actual_task_ids_sha256 != expected_task_ids_sha256:
        raise ValueError(
            "Task-ID SHA-256 mismatch: "
            f"expected {expected_task_ids_sha256}, got {actual_task_ids_sha256}"
        )
    return selected


def replica_count(target: int) -> int:
    if target not in TARGETS:
        raise ValueError(f"Concurrency {target} is not a frozen target")
    if target % BASE_TASK_COUNT:
        raise ValueError("Target concurrency must be divisible by the base task count")
    return target // BASE_TASK_COUNT


def _validate_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")


def _validate_timestamp(timestamp: str) -> None:
    if not _TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError("Official timestamp must use YYYYMMDD_HHmmss")


def _normalize_unique_task_ids(task_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(task_id) for task_id in task_ids)
    if not normalized:
        raise ValueError("At least one task ID is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Task IDs must be unique within a replica")
    return normalized


def build_official_replica_command(
    backend: str,
    timestamp: str,
    task_ids: Sequence[str] = EXPECTED_TASK_IDS,
) -> list[str]:
    """Build one isolated official rollout command without changing upstream."""
    _validate_backend(backend)
    _validate_timestamp(timestamp)
    normalized_task_ids = _normalize_unique_task_ids(task_ids)
    command = [
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
        backend,
        "--split",
        "All",
        "--mode",
        "specific",
        "--task-ids",
        *normalized_task_ids,
        "--no-group-by-site",
        "--concurrency",
        str(PER_REPLICA_CONCURRENCY),
        "--timestamp",
        timestamp,
    ]
    if backend == "local":
        return ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", *command]
    return command


def build_stress_manifest(
    backend: str,
    target: int,
    timestamps: Sequence[str],
    task_ids: Sequence[str] = EXPECTED_TASK_IDS,
) -> dict[str, Any]:
    """Build the secret-free, deterministic manifest for one stress cell."""
    _validate_backend(backend)
    expected_replicas = replica_count(target)
    normalized_timestamps = tuple(str(timestamp) for timestamp in timestamps)
    if len(normalized_timestamps) != expected_replicas:
        raise ValueError(
            f"Replica timestamp count mismatch: expected {expected_replicas}, "
            f"got {len(normalized_timestamps)}"
        )
    if len(set(normalized_timestamps)) != len(normalized_timestamps):
        raise ValueError("Replica timestamps must be unique")
    for timestamp in normalized_timestamps:
        _validate_timestamp(timestamp)

    normalized_task_ids = _normalize_unique_task_ids(task_ids)
    if normalized_task_ids != EXPECTED_TASK_IDS:
        raise ValueError("Stress cells must use the frozen 20-task sample")

    return {
        "schema_version": 1,
        "seed": RANDOM_SEED,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "task_ids": list(normalized_task_ids),
        "task_ids_sha256": task_ids_sha256(normalized_task_ids),
        "backend": backend,
        "target_concurrency": target,
        "per_replica_concurrency": PER_REPLICA_CONCURRENCY,
        "replica_count": expected_replicas,
        "replicas": [
            {"index": index, "timestamp": timestamp}
            for index, timestamp in enumerate(normalized_timestamps)
        ],
    }


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config_snapshot(
    snapshot: Mapping[str, Any],
    *,
    backend: str,
    timestamp: str,
    task_ids: Sequence[str] = EXPECTED_TASK_IDS,
) -> None:
    """Reject a replica whose official redacted snapshot drifted from protocol."""
    _validate_backend(backend)
    _validate_timestamp(timestamp)
    expected_task_ids = list(_normalize_unique_task_ids(task_ids))

    run = snapshot.get("run")
    runtime = snapshot.get("runtime_config")
    if not isinstance(run, Mapping):
        raise ValueError("Missing config snapshot section: run")
    if not isinstance(runtime, Mapping):
        raise ValueError("Missing config snapshot section: runtime_config")

    expected_run: dict[str, Any] = {
        "agent": "browser-use",
        "benchmark": "LexBench-Browser",
        "split": "All",
        "model_id": "qwen3_8B",
        "timestamp": timestamp,
        "model_name_override": "qwen3-8B",
        "browser_id_override": backend,
        "mode": "specific",
        "task_ids": expected_task_ids,
        "concurrency": PER_REPLICA_CONCURRENCY,
    }
    expected_runtime: dict[str, Any] = {
        "browser_id": backend,
        "model_id": "qwen3_8B",
        "max_steps": 40,
        "timeout": 600,
        "flash_mode": True,
        "use_vision": False,
        "use_judge": False,
        "dont_force_structured_output": False,
        "add_schema_to_system_prompt": True,
    }

    mismatches: list[str] = []
    for key, expected in expected_run.items():
        actual = run.get(key)
        if key == "task_ids" and isinstance(actual, Sequence) and not isinstance(actual, str):
            actual = [str(task_id) for task_id in actual]
        if actual != expected:
            mismatches.append(f"run.{key}: expected {expected!r}, got {actual!r}")
    for key, expected in expected_runtime.items():
        actual = runtime.get(key)
        if actual != expected:
            mismatches.append(f"runtime_config.{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ValueError("Config snapshot mismatch: " + "; ".join(mismatches))
