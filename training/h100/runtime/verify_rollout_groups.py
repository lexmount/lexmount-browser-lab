#!/usr/bin/env python3
"""Fail when Verl does not preserve one UID across a task's GRPO rollouts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def split_transfer_queue_key(key: str) -> tuple[str, str]:
    fields = key.rsplit("_", 2)
    if len(fields) != 3:
        raise ValueError(f"unexpected TransferQueue key: {key!r}")
    uid, session_id, _output_index = fields
    return uid, session_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-step", type=int, default=8)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    args = parser.parse_args()

    files = sorted(args.rollout_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"GRPO_GROUP_AUDIT_FAILED no rollout JSONL in {args.rollout_dir}")

    checked_steps = 0
    for path in files:
        sessions_by_uid: dict[str, set[str]] = defaultdict(set)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                uid, session_id = split_transfer_queue_key(str(record["uid"]))
                sessions_by_uid[uid].add(session_id)

        group_sizes = sorted(len(sessions) for sessions in sessions_by_uid.values())
        expected = [args.rollouts_per_task] * args.tasks_per_step
        if group_sizes != expected:
            raise SystemExit(
                "GRPO_GROUP_AUDIT_FAILED "
                f"file={path} tasks={len(group_sizes)} group_sizes={group_sizes} expected={expected}"
            )
        checked_steps += 1

    print(
        "GRPO_GROUP_AUDIT_OK "
        f"steps={checked_steps} tasks_per_step={args.tasks_per_step} "
        f"rollouts_per_task={args.rollouts_per_task}"
    )


if __name__ == "__main__":
    main()
