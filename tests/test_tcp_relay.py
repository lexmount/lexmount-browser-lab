from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "shared-egress-proxy" / "tcp_relay.py"
SPEC = importlib.util.spec_from_file_location("tcp_relay", MODULE_PATH)
assert SPEC and SPEC.loader
tcp_relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tcp_relay)


def test_parse_endpoint_accepts_loopback_upstream() -> None:
    assert tcp_relay.parse_endpoint("127.0.0.1:18781", loopback_only=True) == (
        "127.0.0.1",
        18781,
    )


@pytest.mark.parametrize("value", ["10.0.0.1:18781", "relay.example:18781", "127.0.0.1:0"])
def test_parse_endpoint_rejects_unsafe_upstream(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        tcp_relay.parse_endpoint(value, loopback_only=True)
