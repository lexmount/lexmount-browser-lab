#!/usr/bin/env python3
"""Render deterministic, action-bearing browser parity tasks for a public base URL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_TASKS = 50


def record_for(index: int) -> tuple[str, str, str]:
    if not 1 <= index <= MAX_TASKS:
        raise ValueError(f"task index must be in [1, {MAX_TASKS}]")
    sku = f"PRD-{index:04d}"
    price = f"${10 + ((index * 7) % 89)}.{(index * 13) % 100:02d}"
    stock = f"{4 + ((index * 11) % 53)} units"
    return sku, price, stock


def task_for(index: int, base_url: str) -> dict[str, object]:
    sku, price, stock = record_for(index)
    return {
        "task_id": f"catalog-{index:03d}",
        "question": (
            f"Use the Catalog Lookup page to search product SKU {sku}. "
            "Report its exact listed price and current stock count."
        ),
        "start_url": base_url,
        "website": "ControlledCatalog",
        "split": "controlled-browser-parity-v1",
        "expected_answer": {"must_include": [price, stock], "minimum_act_events": 2},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=MAX_TASKS)
    args = parser.parse_args(argv)
    if not 1 <= args.count <= MAX_TASKS:
        parser.error(f"--count must be in [1, {MAX_TASKS}]")
    base_url = args.base_url.rstrip("/") + "/"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        json.dumps(task_for(index, base_url), ensure_ascii=False) + "\n"
        for index in range(1, args.count + 1)
    )
    args.output.write_text(
        "".join(rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
