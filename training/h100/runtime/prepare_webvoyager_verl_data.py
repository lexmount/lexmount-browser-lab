#!/usr/bin/env python3
"""Convert WebVoyager tasks into task-only Verl agent-loop parquet data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM = """You are an autonomous browser agent. A real browser is open on the task website. First call browser with operation=observe. Use the returned [data-lex-id=lex-N] selectors for grounded actions. Call browser with operation=act and an instruction such as `fill [data-lex-id=lex-0] :: text` or `click [data-lex-id=lex-1]`. Use one browser action per turn, inspect the resulting page, and only then provide a concise final answer supported by browser evidence."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = []
    for line in args.source.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        task_id = str(item["id"])
        rows.append(
            {
                "data_source": "lexbrowser_webvoyager",
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": item["ques"]},
                ],
                "reward_model": {"ground_truth": ""},
                "agent_name": "lexbrowser_tool_agent",
                "extra_info": {
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "browser": {
                            "create_kwargs": {
                                "question": item["ques"],
                                "start_url": item["web"],
                                "task_id": task_id,
                            }
                        }
                    },
                },
            }
        )
        if args.limit and len(rows) >= args.limit:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(json.dumps({
        "rows": len(rows),
        "reference_answers_used": False,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }))


if __name__ == "__main__":
    main()
