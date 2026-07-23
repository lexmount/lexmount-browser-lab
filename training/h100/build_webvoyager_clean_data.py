#!/usr/bin/env python3
"""Rebuild the 168-task webvoyager-clean training set used by the validated run.

Derivation chain (every step is content-addressed):

1. Source: the 600-task cleaned WebVoyager list vendored in this subtree at
   runtime/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/
   datasets/WebVoyager_data_clean.jsonl
   (SHA256 b901adc3f1fb93c069260e1940c59b214374f0ffe58ff7dcf5b1af831d3b1097).
2. Filter: keep only the four stable sites used by the validated 60-step run:
   ArXiv, BBC News, Coursera, GitHub. The result is byte-identical to the
   task manifest of the validated run:
   training/h100/data/webvoyager-clean/tasks.jsonl
   (168 tasks, SHA256
   db0dd8c1f7b2521152caa7dbf76b28dcffa4570655f600007d3f78bd0c8727a9).
3. Parquet: convert tasks.jsonl with the vendored converter
   runtime/prepare_webvoyager_verl_data.py (task-only rows, no
   reference answers). Parquet *bytes* depend on the local pyarrow version;
   the row content is deterministic. The parquet used by the validated
   2026-07-21 run had SHA256
   cc5d673df3700c37eb8439dbe2a649e2309d9c26ccb2c6288507a81c3af24cbb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

H100_ROOT = Path(__file__).resolve().parent
SOURCE_JSONL = (
    H100_ROOT
    / "runtime/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot"
    / "datasets/WebVoyager_data_clean.jsonl"
)
CONVERTER = H100_ROOT / "runtime/prepare_webvoyager_verl_data.py"
ALLOWED_SITES = ("ArXiv", "BBC News", "Coursera", "GitHub")
# Expected hashes live in data/webvoyager-clean/MANIFEST.json (in this subtree)
# so that the data
# contract has a single authoritative location.


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=H100_ROOT / "data/webvoyager-clean",
        help="Directory that receives tasks.jsonl and train.lexbrowser.parquet",
    )
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = H100_ROOT / "data/webvoyager-clean/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_sha = sha256_of(SOURCE_JSONL)
    if source_sha != manifest["source_jsonl_sha256"]:
        sys.exit(f"Source jsonl SHA mismatch: {source_sha}")

    kept = [
        line
        for line in SOURCE_JSONL.read_text(encoding="utf-8").splitlines(keepends=True)
        if json.loads(line)["web_name"] in ALLOWED_SITES
    ]
    tasks_path = out_dir / "tasks.jsonl"
    tasks_path.write_text("".join(kept), encoding="utf-8")
    tasks_sha = sha256_of(tasks_path)
    if tasks_sha != manifest["tasks_jsonl_sha256"]:
        sys.exit(f"tasks.jsonl SHA mismatch: {tasks_sha}")

    parquet_path = out_dir / "train.lexbrowser.parquet"
    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--source",
            str(tasks_path),
            "--output",
            str(parquet_path),
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "tasks": len(kept),
                "tasks_jsonl_sha256": tasks_sha,
                "parquet": str(parquet_path),
                "parquet_sha256": sha256_of(parquet_path),
                "validated_run_parquet_sha256": manifest["validated_run_parquet_sha256"],
                "note": (
                    "Parquet bytes depend on the local pyarrow version; row content "
                    "is deterministic and is what training consumes."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
