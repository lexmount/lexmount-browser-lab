#!/usr/bin/env python3
"""Backend-only Lexmount CDP real-site reachability preflight.

This intentionally bypasses Stagehand so it can distinguish a Stagehand API
navigation failure from a Lexmount Chrome/CDP failure.  It prints no CDP URL
or credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

import websocket
from lexmount import Lexmount


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.google.com/maps/")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--official-proxy", action="store_true")
    parser.add_argument("--region", default=os.environ.get("LEXMOUNT_REGION"))
    parser.add_argument("--browser-mode", choices=("normal", "light"), default="normal")
    args = parser.parse_args()

    client_kwargs: dict[str, str] = {
        "api_key": os.environ["LEXMOUNT_API_KEY"],
        "project_id": os.environ["LEXMOUNT_PROJECT_ID"],
    }
    if args.region:
        client_kwargs["region"] = args.region
    client = Lexmount(**client_kwargs)
    proxy_server = os.environ.get("LEXMOUNT_EXTERNAL_PROXY_SERVER", "").strip()
    if proxy_server:
        username = os.environ.get("LEXMOUNT_EXTERNAL_PROXY_USERNAME", "").strip()
        password = os.environ.get("LEXMOUNT_EXTERNAL_PROXY_PASSWORD", "").strip()
        if not username or not password:
            raise SystemExit(
                "LEXMOUNT_EXTERNAL_PROXY_SERVER requires username and password"
            )
        session_kwargs: dict[str, Any] = {
            "browser_mode": args.browser_mode,
            "proxy": {
                "type": "external", "server": proxy_server,
                "username": username, "password": password,
            },
        }
        proxy_mode = "external"
    else:
        session_kwargs = {
            "browser_mode": args.browser_mode, "official_proxy": args.official_proxy,
        }
        proxy_mode = "official" if args.official_proxy else "direct"
    browser = client.sessions.create(**session_kwargs)
    ws: websocket.WebSocket | None = None
    request_id = 0

    def call(method: str, params: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
        nonlocal request_id
        assert ws is not None
        request_id += 1
        payload: dict[str, Any] = {
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        if session_id:
            payload["sessionId"] = session_id
        ws.send(json.dumps(payload))
        while True:
            response = json.loads(ws.recv())
            if response.get("id") == request_id:
                if "error" in response:
                    raise RuntimeError(f"CDP {method} failed: {response['error']}")
                return response["result"]

    try:
        ws = websocket.create_connection(
            browser.connect_url,
            timeout=args.timeout,
            http_proxy_host=None,
            http_proxy_port=None,
        )
        page_target = call("Target.createTarget", {"url": "about:blank"})["targetId"]
        page_session = call(
            "Target.attachToTarget", {"targetId": page_target, "flatten": True}
        )["sessionId"]
        call("Page.enable", session_id=page_session)
        try:
            call("Page.navigate", {"url": args.url}, session_id=page_session)
        except Exception as exc:
            # A remote Chrome may accept CDP attachment yet never answer the
            # navigation command when its egress path is unavailable.  Keep
            # this as a machine-readable reachability result rather than a
            # Python websocket traceback that obscures the real diagnosis.
            print(
                "LEXMOUNT_CDP_REACHABILITY_FAILED "
                f"mode={proxy_mode} browser={args.browser_mode} reason=cdp_navigation_{type(exc).__name__}"
            )
            raise SystemExit(5) from exc
        # ``Page.navigate`` only confirms that Chrome accepted the request; it
        # can still subsequently render chrome-error://.  Poll rendered state
        # so a preflight cannot falsely pass on ERR_TUNNEL_CONNECTION_FAILED.
        deadline = time.monotonic() + args.timeout
        url = ""
        text = ""
        title = ""
        while time.monotonic() < deadline:
            try:
                snapshot = call(
                    "Runtime.evaluate",
                    {"expression": "JSON.stringify({url:location.href,title:document.title||'',text:(document.body?.innerText||'').slice(0,1000)})", "returnByValue": True},
                    session_id=page_session,
                )
                value = snapshot.get("result", {}).get("value", "{}")
                parsed = json.loads(value or "{}")
                url = str(parsed.get("url", ""))
                title = str(parsed.get("title", ""))
                text = str(parsed.get("text", ""))
                # A URL transition alone is insufficient: Cloudflare and a
                # page still loading both have a final URL but no usable DOM.
                if url and url != "about:blank" and len(text.strip()) >= 20:
                    break
            except (RuntimeError, json.JSONDecodeError):
                pass
            time.sleep(0.5)
        error = re.search(r"\\bERR_[A-Z_]+\\b", text)
        if url.startswith("chrome-error://") or error:
            print(f"LEXMOUNT_CDP_REACHABILITY_FAILED mode={proxy_mode} browser={args.browser_mode} reason={error.group(0) if error else 'chrome-error'}")
            raise SystemExit(2)
        if not url or url == "about:blank" or len(text.strip()) < 20:
            print(f"LEXMOUNT_CDP_REACHABILITY_FAILED mode={proxy_mode} browser={args.browser_mode} reason=empty_or_loading_dom")
            raise SystemExit(3)
        challenge = re.search(
            r"cloudflare|verify you are human|checking your browser|access denied|just a moment",
            f"{title} {text}",
            flags=re.IGNORECASE,
        )
        if challenge:
            print(f"LEXMOUNT_CDP_REACHABILITY_FAILED mode={proxy_mode} browser={args.browser_mode} reason=anti_bot_challenge")
            raise SystemExit(4)
        print(f"LEXMOUNT_CDP_REACHABILITY_OK mode={proxy_mode} browser={args.browser_mode} url={url[:160]} text_chars={len(text)}")
    finally:
        if ws is not None:
            ws.close()
        browser.close()


if __name__ == "__main__":
    main()
