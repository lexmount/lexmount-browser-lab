#!/usr/bin/env python3
"""Fail closed unless independently collected Lexmount and local runs match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "metrics" / "resources_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return manifest, summary


def contract(manifest: dict) -> dict:
    value = manifest.get("comparison_contract", {})
    return json.loads(value) if isinstance(value, str) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexmount-run", type=Path, required=True)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lexmount, lexmount_metrics = load(args.lexmount_run)
    local, local_metrics = load(args.local_run)
    mismatches: list[str] = []
    if lexmount.get("backend") != "lexmount":
        mismatches.append("lexmount manifest backend must be lexmount")
    if local.get("backend") != "local":
        mismatches.append("local manifest backend must be local")
    for label, left, right in (
        ("comparison_contract", contract(lexmount), contract(local)),
        ("topology", lexmount.get("topology"), local.get("topology")),
        ("model.id", lexmount.get("model", {}).get("id"), local.get("model", {}).get("id")),
        (
            "model.resolved_revision",
            lexmount.get("model", {}).get("resolved_revision"),
            local.get("model", {}).get("resolved_revision"),
        ),
        (
            "dataset_sha256",
            lexmount.get("source", {}).get("dataset_sha256"),
            local.get("source", {}).get("dataset_sha256"),
        ),
    ):
        if left != right:
            mismatches.append(f"{label} differs")
    keys = ("last_avg_reward", "last_loss")
    deltas = {}
    for key in keys:
        left, right = (
            lexmount_metrics.get("training", {}).get(key),
            local_metrics.get("training", {}).get(key),
        )
        deltas[key] = None if left is None or right is None else left - right
    result = {
        "schema_version": 1,
        "comparable": not mismatches,
        "mismatches": mismatches,
        "metric_deltas_lexmount_minus_local": deltas,
        "lexmount_run": str(args.lexmount_run),
        "local_run": str(args.local_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
