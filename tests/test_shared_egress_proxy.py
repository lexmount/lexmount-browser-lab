"""Focused contract tests for the shared-egress proxy chain."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def load_proxy_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "shared-egress-proxy"
        / "forward_proxy.py"
    )
    spec = importlib.util.spec_from_file_location("shared_egress_forward_proxy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_upstream_proxy_server_requires_plain_credential_free_http_url():
    proxy = load_proxy_module()

    assert proxy.parse_upstream_proxy_server("http://proxy.example:19002") == (
        "proxy.example",
        19002,
    )
    with pytest.raises(argparse.ArgumentTypeError, match="must not embed credentials"):
        proxy.parse_upstream_proxy_server("http://user:password@proxy.example:19002")
    with pytest.raises(argparse.ArgumentTypeError, match="absolute http"):
        proxy.parse_upstream_proxy_server("https://proxy.example:19002")
    with pytest.raises(argparse.ArgumentTypeError, match="path"):
        proxy.parse_upstream_proxy_server("http://proxy.example:19002/forward")


def test_upstream_proxy_request_keeps_absolute_target_and_replaces_client_auth():
    proxy = load_proxy_module()
    upstream = proxy.UpstreamProxyConfig(
        host="proxy.example",
        port=19002,
        username="user",
        password="pass",
    )

    host, port, request = proxy.upstream_proxy_request(
        "GET",
        "http://example.com/a?b=c",
        "HTTP/1.1",
        {
            "host": "example.com",
            "proxy-authorization": "Basic client-token",
            "connection": "keep-alive",
        },
        upstream,
    )

    assert (host, port) == ("example.com", 80)
    decoded = request.decode("iso-8859-1")
    assert decoded.startswith("GET http://example.com/a?b=c HTTP/1.1\r\n")
    assert "Proxy-Authorization: Basic dXNlcjpwYXNz\r\n" in decoded
    assert "client-token" not in decoded
    assert "Connection: close\r\n" in decoded


def test_upstream_proxy_connect_request_authenticates_the_upstream_only():
    proxy = load_proxy_module()
    upstream = proxy.UpstreamProxyConfig(
        host="proxy.example",
        port=19002,
        username="user",
        password="pass",
    )

    request = proxy.upstream_proxy_connect_request("example.com", 443, upstream).decode(
        "iso-8859-1"
    )

    assert request.startswith("CONNECT example.com:443 HTTP/1.1\r\n")
    assert "Host: example.com:443\r\n" in request
    assert "Proxy-Authorization: Basic dXNlcjpwYXNz\r\n" in request
