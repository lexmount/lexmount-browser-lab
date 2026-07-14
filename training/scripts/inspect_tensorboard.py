#!/usr/bin/env python3
"""Print the latest scalar values from a NeMo-RL TensorBoard run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    events = EventAccumulator(str(args.run), size_guidance={"scalars": 0})
    events.Reload()
    all_tags = events.Tags()
    tags = all_tags["scalars"]
    category_summary = ", ".join(
        f"{name}:{len(values) if hasattr(values, '__len__') else values}"
        for name, values in all_tags.items()
    )
    print(f"event_categories={{{category_summary}}}")
    keywords = (
        "reward",
        "loss",
        "lr",
        "learning_rate",
        "grad",
        "valid",
        "tool_calls",
        "assistant_turns",
        "generation_tokens",
        "infrastructure",
        "timeout",
    )
    print(f"scalar_tag_count={len(tags)}")
    for tag in sorted(tags):
        if not args.all and not any(keyword in tag.lower() for keyword in keywords):
            continue
        values = [(item.step, round(item.value, 8)) for item in events.Scalars(tag)[-args.tail :]]
        print(f"{tag}: {values}")


if __name__ == "__main__":
    main()
