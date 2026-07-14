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

# The upstream task text describes a web task but does not, by itself, tell a
# general instruction-tuned model that browser tools are available and required.
# This prompt is task-agnostic: it does not add demonstrations, answers, or
# synthetic data; it only establishes the BrowserEnv tool-use contract.
BROWSER_AGENT_SYSTEM_PROMPT = """You are an autonomous browser agent operating a real website.
Complete the user's task by using the provided browser tools. Do not claim that you cannot browse the web and do not answer from prior knowledge.
Start by inspecting the page with a browser tool, then use browser observations and actions to make progress. Continue calling tools until you have browser evidence that the task is complete. Only then give a concise final answer based on that evidence."""


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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start offset for an auditable real-task smoke subset (default: 0).",
    )
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

    if args.offset < 0 or args.offset >= len(source_rows):
        raise SystemExit(f"offset must be in [0, {len(source_rows) - 1}], got {args.offset}")
    selected = source_rows[args.offset :] if args.limit < 0 else source_rows[args.offset : args.offset + args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            # Verifiers resolves an environment task by ``task_idx`` against
            # the environment's canonical 600-row dataset.  Preserve that
            # source index for smoke subsets; re-enumerating from zero would
            # silently run Allrecipes--0 when the JSONL prompt says Apple--N.
            source_index = args.offset + index
            prompt = [
                {"role": "system", "content": BROWSER_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": row["ques"]},
            ]
            output_row = {
                "task_idx": source_index,
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
                "task": row["ques"],
                "example_id": source_index,
                "info": {
                    "task_id": row["id"],
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
        "source_offset": args.offset,
        "unique_task_ids": len(set(task_ids)),
        "website_counts": dict(sorted(Counter(row["web_name"] for row in source_rows).items())),
        "environment_id": ENV_ID,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
