from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


forward_proxy = load_module(
    "forward_proxy_half_close_test",
    ROOT / "experiments" / "shared-egress-proxy" / "forward_proxy.py",
)
tcp_relay = load_module(
    "tcp_relay_half_close_test",
    ROOT / "experiments" / "shared-egress-proxy" / "tcp_relay.py",
)


async def response_after_eof(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    payload = await reader.read()
    writer.write(b"response:" + payload)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def client_round_trip(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"request")
    await writer.drain()
    assert writer.can_write_eof()
    writer.write_eof()
    response = await asyncio.wait_for(reader.read(), timeout=2)
    writer.close()
    await writer.wait_closed()
    return response


async def assert_forward_proxy_bridge_preserves_half_close() -> None:
    upstream = await asyncio.start_server(response_after_eof, "127.0.0.1", 0)
    upstream_port = upstream.sockets[0].getsockname()[1]

    async def proxy_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", upstream_port)
        try:
            await forward_proxy.bridge(reader, writer, upstream_reader, upstream_writer)
        finally:
            upstream_writer.close()
            await upstream_writer.wait_closed()
            writer.close()
            await writer.wait_closed()

    proxy = await asyncio.start_server(proxy_handler, "127.0.0.1", 0)
    try:
        proxy_port = proxy.sockets[0].getsockname()[1]
        assert await client_round_trip(proxy_port) == b"response:request"
    finally:
        proxy.close()
        upstream.close()
        await proxy.wait_closed()
        await upstream.wait_closed()


async def assert_tcp_relay_preserves_half_close() -> None:
    upstream = await asyncio.start_server(response_after_eof, "127.0.0.1", 0)
    upstream_port = upstream.sockets[0].getsockname()[1]
    slots = asyncio.Semaphore(1)
    relay = await asyncio.start_server(
        lambda reader, writer: tcp_relay.relay(
            reader,
            writer,
            upstream=("127.0.0.1", upstream_port),
            slots=slots,
        ),
        "127.0.0.1",
        0,
    )
    try:
        relay_port = relay.sockets[0].getsockname()[1]
        assert await client_round_trip(relay_port) == b"response:request"
    finally:
        relay.close()
        upstream.close()
        await relay.wait_closed()
        await upstream.wait_closed()


def test_forward_proxy_bridge_preserves_half_close() -> None:
    asyncio.run(assert_forward_proxy_bridge_preserves_half_close())


def test_tcp_relay_preserves_half_close() -> None:
    asyncio.run(assert_tcp_relay_preserves_half_close())


def test_loopback_is_the_only_private_upstream_exception() -> None:
    assert forward_proxy.is_loopback_literal("127.0.0.1")
    assert forward_proxy.is_loopback_literal("::1")
    assert not forward_proxy.is_loopback_literal("10.0.0.1")
    assert not forward_proxy.is_loopback_literal("relay.example")
