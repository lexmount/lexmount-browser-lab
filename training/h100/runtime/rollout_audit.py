"""Plain JSONL logging for Judge LLM inputs and outputs."""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_judge_io(record: dict[str, Any]) -> str:
    audit_dir = os.environ.get("LEXBROWSER_AUDIT_DIR", "").strip()
    if not audit_dir:
        return ""
    path = Path(audit_dir) / "judge_io.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"logged_at": utc_timestamp(), **record}
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return str(path)
