#!/usr/bin/env python3
"""Fail closed unless every allocated node satisfies the requested GPU preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--expected-gpus-per-node", type=int, required=True)
    parser.add_argument("--expected-gpu-family", required=True)
    args = parser.parse_args()
    rows = []
    errors: list[str] = []
    for path in sorted(args.nodes_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as error:
            errors.append(f"invalid JSON {path}: {error}")
    hosts = [str(row.get("hostname", "")) for row in rows]
    if len(rows) != args.expected_nodes:
        errors.append(f"expected {args.expected_nodes} node reports, found {len(rows)}")
    if len(set(hosts)) != len(hosts):
        errors.append("duplicate hostname in preflight reports")
    gpu_uuids: list[str] = []
    for row in rows:
        host = row.get("hostname", "unknown")
        checks = row.get("checks", {})
        for check in ("gpu_count", "gpu_family", "gpu_memory", "shared_storage_writable"):
            if checks.get(check) is not True:
                errors.append(f"{host}: {check} failed")
        gpus = row.get("gpus", [])
        if len(gpus) != args.expected_gpus_per_node:
            errors.append(f"{host}: expected {args.expected_gpus_per_node} GPUs, found {len(gpus)}")
        gpu_uuids.extend(str(gpu.get("uuid", "")) for gpu in gpus)
    known_uuids = [item for item in gpu_uuids if item]
    if len(set(known_uuids)) != len(known_uuids):
        errors.append("duplicate GPU UUID across node reports")
    result = {
        "schema_version": 1,
        "expected_nodes": args.expected_nodes,
        "expected_gpus_per_node": args.expected_gpus_per_node,
        "expected_gpu_family": args.expected_gpu_family,
        "observed_nodes": len(rows),
        "observed_gpus": len(known_uuids),
        "hosts": hosts,
        "passed": not errors,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("GPU preflight failed:\n- " + "\n- ".join(errors))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
