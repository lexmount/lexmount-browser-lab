#!/usr/bin/env python3
"""Validate the upstream BrowserEnv data and emit NeMo Gym JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_ROWS = 600
EXPECTED_SHA256 = "b901adc3f1fb93c069260e1940c59b214374f0ffe58ff7dcf5b1af831d3b1097"
ENV_ID = "lexbrowser-webvoyager-no-anti-bot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    source_hash = sha256(args.source)
    if source_hash != EXPECTED_SHA256:
        raise SystemExit(
            f"dataset SHA256 mismatch: got {source_hash}, expected {EXPECTED_SHA256}"
        )

    source_rows = [json.loads(line) for line in args.source.read_text().splitlines() if line]
    if len(source_rows) != EXPECTED_ROWS:
        raise SystemExit(f"dataset row mismatch: got {len(source_rows)}, expected {EXPECTED_ROWS}")
    task_ids = [row["id"] for row in source_rows]
    if len(set(task_ids)) != EXPECTED_ROWS:
        raise SystemExit("dataset contains duplicate task IDs")

    selected = source_rows if args.limit < 0 else source_rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            prompt = [{"role": "user", "content": row["ques"]}]
            output_row = {
                "task_idx": index,
                "vf_env_id": ENV_ID,
                "responses_create_params": {"input": prompt},
                "agent_ref": {
                    "type": "responses_api_agents",
                    # This is the top-level NeMo Gym config instance name, not
                    # the reusable server implementation directory/name.
                    "name": "lexbrowser_webvoyager",
                },
                "question": row["ques"],
                "answer": "",
                "task": "webvoyager-no-anti-bot",
                "example_id": index,
                "info": {
                    "source_task_id": row["id"],
                    "start_url": row["web"],
                    "website": row["web_name"],
                },
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(args.source),
        "source_sha256": source_hash,
        "source_rows": len(source_rows),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "output_rows": len(selected),
        "unique_task_ids": len(set(task_ids)),
        "website_counts": dict(sorted(Counter(row["web_name"] for row in source_rows).items())),
        "environment_id": ENV_ID,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
