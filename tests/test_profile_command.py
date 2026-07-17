from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

from lexbrowser_eval.resources.cgroup_profiler import (
    _ensure_user_bus_environment,
    percentile,
    summarize_series,
)


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.95) is None


def test_summarize_series() -> None:
    result = summarize_series([1.0, 2.0, 3.0])

    assert result == {"mean": 2.0, "p95": 3.0, "max": 3.0}


def test_user_bus_environment_uses_existing_runtime_socket(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="profile-bus-") as runtime_dir:
        bus_path = Path(runtime_dir) / "bus"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(bus_path))
        try:
            monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
            monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

            _ensure_user_bus_environment()

            assert os.environ["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        finally:
            listener.close()
