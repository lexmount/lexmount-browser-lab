#!/usr/bin/env python3
"""Run a deliberately narrow forward proxy for paired browser experiments.

It supports HTTPS CONNECT and plain HTTP requests, rejects private-network
destinations, and can require HTTP proxy Basic authentication.  The intended
use is two listeners on the same evaluator host: a loopback-only listener for
local Chrome and an authenticated listener behind a short-lived TCP tunnel for
Lexmount.  Both browser arms then share the evaluator host's public egress.

The loopback listener can also chain to an authenticated public upstream proxy.
That lets local Chrome use a credential-free ``--proxy-server`` while a remote
browser and local Chrome exit through one explicit shared proxy.
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
from pathlib import Path
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
    upstream_proxy: UpstreamProxyConfig | None


@dataclass(frozen=True)
class UpstreamProxyConfig:
    host: str
    port: int
    username: str
    password: str


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


def parse_upstream_proxy_server(value: str) -> tuple[str, int]:
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "http" or not parsed.hostname:
        raise argparse.ArgumentTypeError(
            "--upstream-proxy-server must be an absolute http://HOST:PORT URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise argparse.ArgumentTypeError("--upstream-proxy-server must not embed credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            "--upstream-proxy-server must not include a path, query, or fragment"
        )
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--upstream-proxy-server has an invalid port") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("--upstream-proxy-server port must be between 1 and 65535")
    return parsed.hostname, port


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


def is_loopback_literal(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


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


async def open_upstream_proxy_connection(
    upstream: UpstreamProxyConfig,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to a public upstream proxy or an explicit local relay endpoint."""

    if is_loopback_literal(upstream.host):
        return await asyncio.wait_for(
            asyncio.open_connection(upstream.host, upstream.port), timeout=CONNECT_TIMEOUT_SECONDS
        )
    return await open_public_connection(upstream.host, upstream.port)


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
            f"HTTP/1.1 {status}\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
        + payload
    )
    await writer.drain()


async def relay(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter,
    *,
    direction: str,
) -> None:
    transferred = 0
    try:
        while block := await source.read(64 * 1024):
            transferred += len(block)
            destination.write(block)
            await destination.drain()
    except ConnectionResetError:
        logging.info("Proxy bridge reset direction=%s bytes=%s", direction, transferred)
        raise
    finally:
        logging.info("Proxy bridge eof direction=%s bytes=%s", direction, transferred)
        # A CONNECT client can half-close after sending its request and still
        # need to receive the upstream response. Propagate that EOF instead of
        # treating it as a reason to tear down the opposite direction.
        if not destination.is_closing() and destination.can_write_eof():
            with contextlib.suppress(ConnectionError, OSError, RuntimeError):
                destination.write_eof()
                await destination.drain()


async def bridge(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    tasks = [
        asyncio.create_task(
            relay(client_reader, upstream_writer, direction="client_to_upstream")
        ),
        asyncio.create_task(
            relay(upstream_reader, client_writer, direction="upstream_to_client")
        ),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
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


def basic_proxy_authorization(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def upstream_proxy_connect_request(
    host: str,
    port: int,
    upstream: UpstreamProxyConfig,
) -> bytes:
    authority = f"{host}:{port}"
    authorization = basic_proxy_authorization(upstream.username, upstream.password)
    return (
        f"CONNECT {authority} HTTP/1.1\r\n"
        f"Host: {authority}\r\n"
        f"Proxy-Authorization: {authorization}\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode("iso-8859-1")


def upstream_proxy_request(
    method: str,
    target: str,
    version: str,
    headers: dict[str, str],
    upstream: UpstreamProxyConfig,
) -> tuple[str, int, bytes]:
    parsed = urlsplit(target)
    if parsed.scheme.lower() != "http" or not parsed.hostname:
        raise ValueError("only absolute http URLs are supported outside CONNECT")
    host, port = parse_authority(parsed.netloc, default_port=80)
    forwarded = [f"{method} {target} {version}"]
    seen_host = False
    for name, value in headers.items():
        if name in {"proxy-authorization", "proxy-connection", "connection"}:
            continue
        if name == "host":
            seen_host = True
        forwarded.append(f"{name.title()}: {value}")
    if not seen_host:
        forwarded.append(f"Host: {parsed.netloc}")
    forwarded.append(
        f"Proxy-Authorization: {basic_proxy_authorization(upstream.username, upstream.password)}"
    )
    forwarded.append("Connection: close")
    return host, port, ("\r\n".join(forwarded) + "\r\n\r\n").encode("iso-8859-1")


async def upstream_proxy_connect(
    upstream: UpstreamProxyConfig,
    host: str,
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
    reader, writer = await open_upstream_proxy_connection(upstream)
    try:
        writer.write(upstream_proxy_connect_request(host, port, upstream))
        await writer.drain()
        response = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"), timeout=CONNECT_TIMEOUT_SECONDS
        )
        if len(response) > MAX_HEADER_BYTES:
            raise ValueError("upstream proxy response headers exceed limit")
        status = response.decode("iso-8859-1").split("\r\n", 1)[0]
        if not status.startswith("HTTP/"):
            raise ConnectionError("upstream proxy returned a malformed response")
        if not status.split(" ", 2)[1:2] == ["200"]:
            raise ConnectionError(f"upstream proxy rejected CONNECT: {status}")
        return reader, writer, response
    except BaseException:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        raise


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
                if config.upstream_proxy is None:
                    upstream_reader, upstream_writer = await open_public_connection(host, port)
                    response = b"HTTP/1.1 200 Connection Established\r\n\r\n"
                else:
                    await resolve_public_addresses(host, port)
                    upstream_reader, upstream_writer, response = await upstream_proxy_connect(
                        config.upstream_proxy, host, port
                    )
                logging.info("CONNECT accepted peer=%s host=%s port=%s", peer, host, port)
                writer.write(response)
                await writer.drain()
                await bridge(reader, writer, upstream_reader, upstream_writer)
                return

            if config.upstream_proxy is None:
                host, port, request = upstream_request(method, target, version, headers)
                upstream_reader, upstream_writer = await open_public_connection(host, port)
            else:
                host, port, request = upstream_proxy_request(
                    method, target, version, headers, config.upstream_proxy
                )
                await resolve_public_addresses(host, port)
                upstream_reader, upstream_writer = await open_upstream_proxy_connection(
                    config.upstream_proxy
                )
            logging.info("HTTP accepted peer=%s host=%s port=%s", peer, host, port)
            upstream_writer.write(request)
            await upstream_writer.drain()
            await bridge(reader, writer, upstream_reader, upstream_writer)
    except asyncio.IncompleteReadError as exc:
        logging.info("Proxy request incomplete peer=%s bytes=%s", peer, len(exc.partial))
        return
    except Exception as exc:
        # Python 3.10 exposes asyncio.TimeoutError separately from the
        # builtin TimeoutError. Keep request failures observable on both
        # evaluator runtimes without swallowing asyncio cancellation.
        logging.info(
            "Proxy request rejected peer=%s error=%s reason=%s",
            peer,
            type(exc).__name__,
            exc,
        )
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
    upstream_values = (
        args.upstream_proxy_server,
        args.upstream_proxy_username,
        args.upstream_proxy_password,
    )
    if any(upstream_values) and not all(upstream_values):
        raise ValueError(
            "--upstream-proxy-server, --upstream-proxy-username, and "
            "--upstream-proxy-password-file must be provided together"
        )
    upstream_proxy = None
    if args.upstream_proxy_server:
        host, port = parse_upstream_proxy_server(args.upstream_proxy_server)
        upstream_proxy = UpstreamProxyConfig(
            host=host,
            port=port,
            username=args.upstream_proxy_username,
            password=args.upstream_proxy_password,
        )
    host, port = args.listen
    config = ProxyConfig(
        username=username,
        password=password,
        max_connections=args.max_connections,
        upstream_proxy=upstream_proxy,
    )
    slots = asyncio.Semaphore(config.max_connections)
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(reader, writer, config=config, slots=slots),
        host=host,
        port=port,
        limit=MAX_HEADER_BYTES,
    )
    auth_mode = "basic-auth" if username else "loopback-no-auth"
    route_mode = "via-upstream" if upstream_proxy else "direct-egress"
    logging.info(
        "paired-egress proxy listening on %s:%s (%s, %s)", host, port, auth_mode, route_mode
    )
    async with server:
        await server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=parse_listen, required=True)
    parser.add_argument("--username")
    password_group = parser.add_mutually_exclusive_group()
    password_group.add_argument("--password")
    password_group.add_argument("--password-file", type=Path)
    parser.add_argument("--upstream-proxy-server")
    parser.add_argument("--upstream-proxy-username")
    upstream_password_group = parser.add_mutually_exclusive_group()
    upstream_password_group.add_argument("--upstream-proxy-password")
    upstream_password_group.add_argument("--upstream-proxy-password-file", type=Path)
    parser.add_argument("--max-connections", type=int, default=MAX_CONNECTIONS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_connections < 1:
        raise SystemExit("--max-connections must be positive")
    if args.password_file is not None:
        try:
            args.password = args.password_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"could not read --password-file: {exc}") from exc
    if args.upstream_proxy_password_file is not None:
        try:
            args.upstream_proxy_password = args.upstream_proxy_password_file.read_text(
                encoding="utf-8"
            ).strip()
        except OSError as exc:
            raise SystemExit(f"could not read --upstream-proxy-password-file: {exc}") from exc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
