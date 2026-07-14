#!/usr/bin/env python3
"""Assert that every requested GPU rank can complete one NCCL all-reduce."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = float(dist.get_world_size())
    if value.item() != expected:
        raise SystemExit(f"NCCL all-reduce produced {value.item()}, expected {expected}")
    if dist.get_rank() == 0:
        run_dir = Path(os.environ["LEXBROWSER_CONTAINER_RUN_DIR"])
        output = run_dir / "preflight" / "nccl.json"
        output.write_text(
            json.dumps(
                {"world_size": dist.get_world_size(), "backend": "nccl", "passed": True}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
