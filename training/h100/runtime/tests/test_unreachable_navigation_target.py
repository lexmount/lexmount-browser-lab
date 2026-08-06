"""A URL the policy invented is the policy's mistake, not an environment failure.

On the 2026-08-06 light run, 14 navigations ended on Chrome's error page and
were all classified `ERROR_INFRASTRUCTURE_NAVIGATE`, which excluded the rollout
from the GRPO group statistics.  Every one of the 14 target hostnames resolves
nowhere — `example-conference-site.com`, `store.arxiv.org`, `www.sfl.scot` —
so the model was being spared the negative reward for URLs it made up.

The engine reports DNS failures as `ERR_FAILED`, not `ERR_NAME_NOT_RESOLVED`,
so the code cannot be the discriminator.  Who chose the URL is.
"""
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "lexbrowser_webvoyager" / "src" / \
    "lexbrowser_webvoyager_no_anti_bot" / "environment.py"


def _load():
    """Import the classifier helpers without the browser stack installed."""
    for name in ("websocket", "lexmount", "nemo_gym", "playwright",
                 "playwright.sync_api"):
        sys.modules.setdefault(name, types.ModuleType(name))
    source = SRC.read_text(encoding="utf-8")
    start = source.index("_POLICY_GROUNDING_ERROR_PATTERNS =")
    end = source.index("@dataclass", start)
    module = types.ModuleType("_env_helpers")
    exec(compile(source[start:end], "environment.py", "exec"), module.__dict__)
    return module._is_unreachable_target_failure


is_unreachable = _load()


def _error_page(code):
    return RuntimeError(f"infrastructure_browser_error_page: {code}")


# --- the policy's own bad URL ----------------------------------------------

def test_err_failed_on_a_chosen_url_is_the_policy_s_mistake():
    assert is_unreachable(_error_page("ERR_FAILED")) is True


def test_name_not_resolved_is_the_policy_s_mistake():
    assert is_unreachable(_error_page("ERR_NAME_NOT_RESOLVED")) is True


def test_connection_refused_is_the_policy_s_mistake():
    assert is_unreachable(_error_page("ERR_CONNECTION_REFUSED")) is True


# --- our transport being down ----------------------------------------------

def test_tunnel_failure_stays_infrastructure():
    # The provider proxy is down; every URL would have failed identically, so
    # blaming the policy for its choice of target would be wrong.
    assert is_unreachable(_error_page("ERR_TUNNEL_CONNECTION_FAILED")) is False


def test_proxy_failure_stays_infrastructure():
    assert is_unreachable(_error_page("ERR_PROXY_CONNECTION_FAILED")) is False


def test_internet_disconnected_stays_infrastructure():
    assert is_unreachable(_error_page("ERR_INTERNET_DISCONNECTED")) is False


# --- everything that is not an error page ----------------------------------

def test_a_timeout_is_not_an_unreachable_target():
    assert is_unreachable(TimeoutError("infrastructure_setup_document_not_ready")) is False


def test_an_anti_bot_challenge_is_not_an_unreachable_target():
    assert is_unreachable(RuntimeError("infrastructure_anti_bot_challenge")) is False


def test_a_cdp_fault_is_not_an_unreachable_target():
    assert is_unreachable(
        RuntimeError("CDP Runtime.evaluate failed: Promise was collected")
    ) is False


def test_a_stale_selector_is_not_an_unreachable_target():
    # Already handled by the grounding classifier; must not be double-counted.
    assert is_unreachable(RuntimeError("Error: selector not found")) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
