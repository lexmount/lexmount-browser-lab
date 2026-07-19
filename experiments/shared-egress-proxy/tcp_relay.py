#!/usr/bin/env python3
"""Relay a public TCP listener to a loopback-only upstream endpoint.

The public listener is intentionally narrow: its upstream must resolve to a
literal loopback address. It is meant to sit in front of an authenticated
proxy that arrives through an SSH reverse tunnel during a paired browser run.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress


def parse_endpoint(value: str, *, loopback_only: bool = False) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host or not raw_port.isdigit():
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("endpoint port must be between 1 and 65535")
    if loopback_only:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("upstream must use a literal loopback IP") from exc
        if not address.is_loopback:
            raise argparse.ArgumentTypeError("upstream must use a loopback IP")
    return host, port


async def copy(source: asyncio.StreamReader, destination: asyncio.StreamWriter) -> None:
    while block := await source.read(64 * 1024):
        destination.write(block)
        await destination.drain()


async def relay(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    upstream: tuple[str, int],
    slots: asyncio.Semaphore,
) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    try:
        async with slots:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(*upstream), timeout=15.0
            )
            tasks = [
                asyncio.create_task(copy(reader, upstream_writer)),
                asyncio.create_task(copy(upstream_reader, writer)),
            ]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except (OSError, TimeoutError):
        pass
    finally:
        if upstream_writer is not None:
            upstream_writer.close()
            with contextlib.suppress(Exception):
                await upstream_writer.wait_closed()
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def serve(args: argparse.Namespace) -> None:
    slots = asyncio.Semaphore(args.max_connections)
    listener = await asyncio.start_server(
        lambda reader, writer: relay(
            reader,
            writer,
            upstream=args.upstream,
            slots=slots,
        ),
        *args.listen,
    )
    async with listener:
        await listener.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--max-connections", type=int, default=64)
    args = parser.parse_args()
    args.listen = parse_endpoint(args.listen)
    args.upstream = parse_endpoint(args.upstream, loopback_only=True)
    if args.max_connections < 1:
        parser.error("--max-connections must be positive")
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
