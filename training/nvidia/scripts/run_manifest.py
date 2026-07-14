#!/usr/bin/env python3
"""Create and update a secret-free manifest for an NVIDIA delivery run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_SECRETS = (
    "LEXMOUNT_BASE_URL",
    "LEXMOUNT_API_KEY",
    "LEXMOUNT_PROJECT_ID",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


def now() -> str:
    # The NeMo v0.6 base image and local delivery checks may use Python 3.9.
    return (
        datetime.now(timezone.utc)  # noqa: UP017
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_env_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].removeprefix("export ").strip())
    return names


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifests" / "run_manifest.json"


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.is_file():
        raise SystemExit(f"run manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def command_init(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    env_names = read_env_names(args.secrets_file)
    try:
        comparison_contract = json.loads(args.comparison_contract)
    except json.JSONDecodeError as error:
        raise SystemExit(f"comparison contract must be JSON: {error}") from error
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": now(),
        "updated_at": now(),
        "status": "created",
        "mode": args.mode,
        "backend": args.backend,
        "topology": {
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "expected_gpus": args.nodes * args.gpus_per_node,
            "gpu_family": args.gpu_family,
        },
        "source": {
            "repository_root": str(args.root.resolve()),
            "git_revision": git_revision(args.root),
            "training_config": str(args.config),
            "training_config_sha256": sha256(args.config),
            "dataset": str(args.dataset),
            "dataset_sha256": sha256(args.dataset),
        },
        "model": {"id": args.model_id, "requested_revision": args.model_revision},
        "secrets": {
            "file_present": args.secrets_file.is_file(),
            "required_names_present": {name: name in env_names for name in REQUIRED_SECRETS},
        },
        "comparison_contract": comparison_contract,
        "resume_from": args.resume_from,
        "phases": [
            {"name": "init", "status": "complete", "at": now(), "detail": "manifest created"}
        ],
    }
    write(manifest_path(run_dir), manifest)
    print(manifest_path(run_dir))


def command_phase(args: argparse.Namespace) -> None:
    manifest = load(args.run_dir)
    manifest["updated_at"] = now()
    manifest["status"] = (
        args.status if args.status in {"failed", "complete"} else manifest.get("status", "created")
    )
    manifest.setdefault("phases", []).append(
        {"name": args.name, "status": args.status, "at": now(), "detail": args.detail}
    )
    write(manifest_path(args.run_dir), manifest)


def command_model(args: argparse.Namespace) -> None:
    manifest = load(args.run_dir)
    manifest["updated_at"] = now()
    manifest.setdefault("model", {}).update(
        {
            "resolved_revision": args.revision,
            "snapshot_sha256": args.snapshot_sha256,
            "path": args.path,
        }
    )
    write(manifest_path(args.run_dir), manifest)


def command_finalize(args: argparse.Namespace) -> None:
    manifest = load(args.run_dir)
    manifest["updated_at"] = now()
    manifest["completed_at"] = now()
    manifest["status"] = args.status
    manifest.setdefault("phases", []).append(
        {"name": "finalize", "status": args.status, "at": now(), "detail": args.detail}
    )
    write(manifest_path(args.run_dir), manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--dataset", type=Path, required=True)
    init.add_argument("--secrets-file", type=Path, required=True)
    init.add_argument("--mode", required=True)
    init.add_argument("--backend", default="lexmount")
    init.add_argument("--nodes", type=int, required=True)
    init.add_argument("--gpus-per-node", type=int, required=True)
    init.add_argument("--gpu-family", required=True)
    init.add_argument("--model-id", required=True)
    init.add_argument("--model-revision", required=True)
    init.add_argument("--comparison-contract", required=True)
    init.add_argument("--resume-from")
    init.set_defaults(func=command_init)
    phase = subparsers.add_parser("phase")
    phase.add_argument("--run-dir", type=Path, required=True)
    phase.add_argument("--name", required=True)
    phase.add_argument(
        "--status", choices=("started", "complete", "failed", "skipped"), required=True
    )
    phase.add_argument("--detail", default="")
    phase.set_defaults(func=command_phase)
    model = subparsers.add_parser("model")
    model.add_argument("--run-dir", type=Path, required=True)
    model.add_argument("--revision", required=True)
    model.add_argument("--snapshot-sha256", required=True)
    model.add_argument("--path", required=True)
    model.set_defaults(func=command_model)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--status", choices=("complete", "failed"), required=True)
    finalize.add_argument("--detail", default="")
    finalize.set_defaults(func=command_finalize)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
