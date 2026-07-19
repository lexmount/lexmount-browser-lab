#!/usr/bin/env python3
"""Extract a valid TCP endpoint from cpolar's mixed text/JSON log output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TCP_ENDPOINT_PATTERN = re.compile(r'tcp://([^\\\s"}]+)')
HOST_PORT_PATTERN = re.compile(r"[A-Za-z0-9._-]+:[0-9]{1,5}")


def extract_endpoint(text: str) -> str:
    """Return the most recent valid cpolar host:port without log escaping."""

    for raw in reversed(TCP_ENDPOINT_PATTERN.findall(text)):
        endpoint = raw.rstrip("/")
        if not HOST_PORT_PATTERN.fullmatch(endpoint):
            continue
        port = int(endpoint.rsplit(":", 1)[1])
        if 1 <= port <= 65535:
            return endpoint
    raise ValueError("no valid cpolar TCP endpoint found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    try:
        endpoint = extract_endpoint(args.log.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not extract cpolar TCP endpoint: {exc}") from exc
    print(endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
