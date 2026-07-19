#!/usr/bin/env python3
"""Run a deliberately narrow forward proxy for paired browser experiments.

It supports HTTPS CONNECT and plain HTTP requests, rejects private-network
destinations, and can require HTTP proxy Basic authentication.  The intended
use is two listeners on the same evaluator host: a loopback-only listener for
local Chrome and an authenticated listener behind a short-lived TCP tunnel for
Lexmount.  Both browser arms then share the evaluator host's public egress.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import ipaddress
import logging
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_HEADER_BYTES = 64 * 1024
MAX_CONNECTIONS = 64
CONNECT_TIMEOUT_SECONDS = 20.0
ALLOWED_PORTS = {80, 443}


@dataclass(frozen=True)
class ProxyConfig:
    username: str | None
    password: str | None
    max_connections: int


def parse_listen(value: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host or not raw_port.isdigit():
        raise argparse.ArgumentTypeError("--listen must be HOST:PORT")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("listen port must be between 1 and 65535")
    return host, port


def parse_authority(value: str, *, default_port: int | None = None) -> tuple[str, int]:
    value = value.strip()
    if value.startswith("["):
        host, marker, suffix = value[1:].partition("]")
        if not marker:
            raise ValueError("invalid bracketed host")
        raw_port = suffix.removeprefix(":")
    else:
        host, marker, raw_port = value.rpartition(":")
        if not marker:
            if default_port is None:
                raise ValueError("target must include a port")
            host, raw_port = value, str(default_port)
    if not host or not raw_port.isdigit():
        raise ValueError("invalid target host or port")
    port = int(raw_port)
    if port not in ALLOWED_PORTS:
        raise ValueError(f"target port {port} is not allowed")
    return host, port


def is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def resolve_public_addresses(host: str, port: int) -> Iterable[tuple[str, int]]:
    lowered = host.rstrip(".").lower()
    if lowered in {"localhost", "localhost.localdomain"} or lowered.endswith(".local"):
        raise ValueError("local destinations are not allowed")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not is_public_address(str(literal)):
            raise ValueError("non-public destination is not allowed")
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        return [(str(literal), family)]

    infos = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    candidates: list[tuple[str, int]] = []
    for family, _, _, _, address in infos:
        ip = str(address[0])
        if is_public_address(ip) and (ip, family) not in candidates:
            candidates.append((ip, family))
    if not candidates:
        raise ValueError("hostname resolved only to non-public addresses")
    return candidates


async def open_public_connection(
    host: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    last_error: BaseException | None = None
    for address, family in await resolve_public_addresses(host, port):
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(address, port, family=family),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, TimeoutError) as exc:
            last_error = exc
    raise ConnectionError(f"could not reach public target: {last_error}")


def request_headers(raw: bytes) -> tuple[str, str, str, dict[str, str]]:
    lines = raw.decode("iso-8859-1").split("\r\n")
    method, target, version = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            break
        name, separator, value = line.partition(":")
        if not separator:
            raise ValueError("malformed request header")
        headers[name.strip().lower()] = value.strip()
    return method.upper(), target, version, headers


def authorized(headers: dict[str, str], config: ProxyConfig) -> bool:
    if config.username is None:
        return True
    raw = headers.get("proxy-authorization", "")
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "basic" or not token:
        return False
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    return decoded == f"{config.username}:{config.password}"


async def write_response(writer: asyncio.StreamWriter, status: str, body: str = "") -> None:
    payload = body.encode("utf-8")
    writer.write(
        (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        + payload
    )
    await writer.drain()


async def relay(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter,
) -> None:
    while block := await source.read(64 * 1024):
        destination.write(block)
        await destination.drain()


async def bridge(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    tasks = [
        asyncio.create_task(relay(client_reader, upstream_writer)),
        asyncio.create_task(relay(upstream_reader, client_writer)),
    ]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def upstream_request(
    method: str,
    target: str,
    version: str,
    headers: dict[str, str],
) -> tuple[str, int, bytes]:
    parsed = urlsplit(target)
    if parsed.scheme.lower() != "http" or not parsed.hostname:
        raise ValueError("only absolute http URLs are supported outside CONNECT")
    host, port = parse_authority(
        parsed.netloc,
        default_port=80,
    )
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    forwarded = [f"{method} {path} {version}"]
    seen_host = False
    for name, value in headers.items():
        if name in {"proxy-authorization", "proxy-connection", "connection"}:
            continue
        if name == "host":
            seen_host = True
        forwarded.append(f"{name.title()}: {value}")
    if not seen_host:
        forwarded.append(f"Host: {parsed.netloc}")
    forwarded.append("Connection: close")
    return host, port, ("\r\n".join(forwarded) + "\r\n\r\n").encode("iso-8859-1")


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    config: ProxyConfig,
    slots: asyncio.Semaphore,
) -> None:
    peer = writer.get_extra_info("peername")
    upstream_writer: asyncio.StreamWriter | None = None
    try:
        async with slots:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15.0)
            if len(raw) > MAX_HEADER_BYTES:
                raise ValueError("request headers exceed limit")
            method, target, version, headers = request_headers(raw)
            if not authorized(headers, config):
                writer.write(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                    b"Proxy-Authenticate: Basic realm=paired-egress\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                return
            if method == "CONNECT":
                host, port = parse_authority(target)
                upstream_reader, upstream_writer = await open_public_connection(host, port)
                logging.info("CONNECT accepted peer=%s host=%s port=%s", peer, host, port)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                await bridge(reader, writer, upstream_reader, upstream_writer)
                return

            host, port, request = upstream_request(method, target, version, headers)
            upstream_reader, upstream_writer = await open_public_connection(host, port)
            logging.info("HTTP accepted peer=%s host=%s port=%s", peer, host, port)
            upstream_writer.write(request)
            await upstream_writer.drain()
            await bridge(reader, writer, upstream_reader, upstream_writer)
    except asyncio.IncompleteReadError:
        return
    except (ConnectionError, OSError, ValueError, TimeoutError) as exc:
        logging.info("Proxy request rejected peer=%s reason=%s", peer, exc)
        with contextlib.suppress(ConnectionError):
            await write_response(writer, "502 Bad Gateway", "proxy target unavailable\n")
    finally:
        if upstream_writer is not None:
            upstream_writer.close()
            with contextlib.suppress(Exception):
                await upstream_writer.wait_closed()
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def serve(args: argparse.Namespace) -> None:
    username = args.username or None
    password = args.password or None
    if bool(username) != bool(password):
        raise ValueError("--username and --password must be provided together")
    host, port = args.listen
    config = ProxyConfig(
        username=username,
        password=password,
        max_connections=args.max_connections,
    )
    slots = asyncio.Semaphore(config.max_connections)
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(reader, writer, config=config, slots=slots),
        host=host,
        port=port,
        limit=MAX_HEADER_BYTES,
    )
    auth_mode = "basic-auth" if username else "loopback-no-auth"
    logging.info("paired-egress proxy listening on %s:%s (%s)", host, port, auth_mode)
    async with server:
        await server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=parse_listen, required=True)
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--max-connections", type=int, default=MAX_CONNECTIONS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_connections < 1:
        raise SystemExit("--max-connections must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
