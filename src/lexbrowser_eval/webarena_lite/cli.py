#!/usr/bin/env python3
"""
One-command runner for glm-5.2 on VAB-WebArena-Lite.

Expected external usage from /data/wf/sxh:

    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://litellm.local.lexmount.net/v1
    export OPENAI_MODEL=glm-5.2
    python run_webarena_lite.py

The script bootstraps the VAB-WebArena-Lite harness, prepares the Python
environment, generates task configs, refreshes login cookies, runs the WebRL
text-mode benchmark, scores it, and writes a compact report.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import venv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

VAB_REPO = "https://github.com/THUDM/VisualAgentBench.git"
VISUALWEBARENA_REPO = "https://github.com/web-arena-x/visualwebarena.git"
VISUALWEBARENA_COMMIT = "ad57aae4dad71531504726900b80db02e0526158"
TASK_COUNT = 165
DEFAULT_SERVER = "10.2.131.41"
WIKI_PATH = "wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
SITE_ENV_KEYS = {
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "MAP",
    "MAP_TILE",
    "WIKIPEDIA",
    "HOMEPAGE",
    "CLASSIFIEDS",
    "CLASSIFIEDS_RESET_TOKEN",
}
WEBRL_ACTION_LOOKUP = 'action_type = action["action"].lower()'
WEBRL_SAFE_ACTION_LOOKUP = 'action_type = str(action.get("action", "")).lower()'
AUTO_LOGIN_PAGE = "    page = context.new_page()"
AUTO_LOGIN_TIMEOUT_BLOCK = (
    f"{AUTO_LOGIN_PAGE}\n"
    "    page.set_default_timeout(120_000)\n"
    "    page.set_default_navigation_timeout(120_000)"
)
REQUIRED_AUTH_COMBINATIONS = (
    ("gitlab",),
    ("reddit",),
    ("shopping_admin",),
    ("shopping",),
    ("gitlab", "reddit"),
)
RUNTIME_REQUIREMENTS = (
    "lxml==4.9.3",
    "dashscope==1.14.1",
    "anthropic==0.4.1",
)


@dataclass
class RunConfig:
    result_dir: pathlib.Path
    test_start_idx: int = 0
    test_end_idx: int = TASK_COUNT
    model: str = "glm-5.2"
    temperature: float = 0.01
    top_p: float = 0.9
    max_tokens: int = 2048
    max_steps: int = 30
    viewport_width: int = 1280
    viewport_height: int = 720
    parsing_failure_th: int = 5
    repeating_action_failure_th: int = 5


def log(message: str) -> None:
    print(f"[webarena-lite] {message}", flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"[webarena-lite] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def run(
    command: Sequence[str],
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    log("$ " + " ".join(command))
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        check=check,
    )


def capture(
    command: Sequence[str],
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def require_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        die(f"Missing required environment variable: {name}")
    return value


def slug_model(model: str) -> str:
    return "".join(ch for ch in model.lower() if ch.isalnum())


def default_result_dir(
    root: pathlib.Path, model: str, now: dt.datetime | None = None
) -> pathlib.Path:
    if now is None:
        now = dt.datetime.now()
    return root / "results" / (f"{slug_model(model)}_webarena_lite_{now.strftime('%Y%m%d_%H%M%S')}")


def default_runtime_root(
    script_root: pathlib.Path,
    exists=os.path.exists,
    writable=os.access,
) -> pathlib.Path:
    data_root = pathlib.Path("/data/wf/sxh")
    try:
        if exists(data_root) and writable(data_root):
            return data_root
    except TypeError:
        if exists(data_root) and writable(data_root, os.W_OK):
            return data_root
    return script_root


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def site_url(environ: Mapping[str, str], name: str, default: str) -> str:
    return environ.get(name, "").strip() or default


def parse_site_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in SITE_ENV_KEYS:
            continue
        value = value.strip().strip("'").strip('"')
        if value:
            values[key] = value
    return values


def environ_with_site_env(runtime_root: pathlib.Path, environ: Mapping[str, str]) -> dict[str, str]:
    merged = dict(environ)
    file_values = parse_site_env(runtime_root / "webarena_env" / "site_env.sh")
    for key, value in file_values.items():
        if merged.get("WEBARENA_SERVER") and key != "CLASSIFIEDS_RESET_TOKEN":
            continue
        if not merged.get(key):
            merged[key] = value
    return merged


def build_harness_env(
    environ: Mapping[str, str],
    server: str,
    map_server: str | None,
) -> dict[str, str]:
    base_url = require_env(environ, "OPENAI_BASE_URL")
    api_key = require_env(environ, "OPENAI_API_KEY")
    model = require_env(environ, "OPENAI_MODEL")
    map_host = map_server or server

    map_url = site_url(environ, "MAP", f"http://{map_host}:3000")
    env = dict(os.environ)
    env.update(
        {
            "DATASET": "webarena",
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_URL": normalize_base_url(base_url),
            "OPENAI_BASE_URL": normalize_base_url(base_url),
            "OPENAI_MODEL": model,
            "TOKENIZERS_PARALLELISM": "false",
            "CLASSIFIEDS": site_url(environ, "CLASSIFIEDS", f"http://{server}:9980"),
            "CLASSIFIEDS_RESET_TOKEN": site_url(
                environ,
                "CLASSIFIEDS_RESET_TOKEN",
                "4b61655535e7ed388f0d40a93600254c",
            ),
            "SHOPPING": site_url(environ, "SHOPPING", f"http://{server}:7770"),
            "SHOPPING_ADMIN": site_url(environ, "SHOPPING_ADMIN", f"http://{server}:7780/admin"),
            "REDDIT": site_url(environ, "REDDIT", f"http://{server}:9999"),
            "GITLAB": site_url(environ, "GITLAB", f"http://{server}:8023"),
            "MAP": map_url,
            "MAP_TILE": site_url(
                environ,
                "MAP_TILE",
                f"{normalize_base_url(map_url)}/tile/10/284/385.png",
            ),
            "WIKIPEDIA": site_url(environ, "WIKIPEDIA", f"http://{server}:8888/{WIKI_PATH}"),
            "HOMEPAGE": site_url(environ, "HOMEPAGE", f"http://{server}:4399"),
        }
    )
    return env


def find_python310() -> str:
    candidates = [
        os.environ.get("PYTHON310", ""),
        "python3.11",
        "python3.10",
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) if os.path.basename(candidate) == candidate else candidate
        if not path:
            continue
        proc = capture(
            [
                path,
                "-c",
                "import sys; raise SystemExit("
                "0 if (3,10) <= sys.version_info[:2] <= (3,11) else 1)",
            ],
            check=False,
        )
        if proc.returncode == 0:
            return path
    die("Python 3.10 or 3.11 is required. Set PYTHON310=/path/to/python3.10.")
    raise AssertionError("unreachable")


def ensure_checkout(work_dir: pathlib.Path) -> pathlib.Path:
    wab_dir = work_dir / "VisualAgentBench"
    harness_dir = wab_dir / "VAB-WebArena-Lite"

    if not wab_dir.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", VAB_REPO, str(wab_dir)])
    elif not (wab_dir / ".git").exists():
        die(f"{wab_dir} exists but is not a git checkout")

    if not harness_dir.exists():
        die(f"Expected VAB-WebArena-Lite directory was not found in {wab_dir}")

    if not (harness_dir / "run.py").exists():
        visual_dir = harness_dir / "visualwebarena"
        if not visual_dir.exists():
            run(
                ["git", "clone", "--depth", "1", VISUALWEBARENA_REPO, "visualwebarena"],
                cwd=harness_dir,
            )
        run(["git", "reset", "--hard", VISUALWEBARENA_COMMIT], cwd=visual_dir)
        run(["bash", "replace.sh"], cwd=harness_dir)

    return harness_dir


def venv_python(venv_dir: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def patch_webrl_action_parser(harness_dir: pathlib.Path) -> None:
    actions_path = harness_dir / "browser_env" / "actions.py"
    source = actions_path.read_text(encoding="utf-8")
    if WEBRL_SAFE_ACTION_LOOKUP in source:
        return
    if WEBRL_ACTION_LOOKUP not in source:
        die(f"Cannot locate WebRL action lookup in {actions_path}")
    actions_path.write_text(
        source.replace(WEBRL_ACTION_LOOKUP, WEBRL_SAFE_ACTION_LOOKUP, 1),
        encoding="utf-8",
    )


def patch_auto_login_timeout(harness_dir: pathlib.Path) -> None:
    auto_login_path = harness_dir / "browser_env" / "auto_login.py"
    source = auto_login_path.read_text(encoding="utf-8")
    if AUTO_LOGIN_TIMEOUT_BLOCK in source:
        return
    page_count = source.count(AUTO_LOGIN_PAGE)
    if page_count != 2:
        die(f"Expected two login page initializers in {auto_login_path}, found {page_count}")
    auto_login_path.write_text(
        source.replace(AUTO_LOGIN_PAGE, AUTO_LOGIN_TIMEOUT_BLOCK),
        encoding="utf-8",
    )


def runtime_dependencies_ready(
    python_bin: pathlib.Path, runtime_env: Mapping[str, str]
) -> bool:
    probe = textwrap.dedent(
        """
        import pathlib

        import anthropic
        import dashscope
        import lxml
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        try:
            executable = pathlib.Path(playwright.chromium.executable_path)
            raise SystemExit(0 if executable.is_file() else 1)
        finally:
            playwright.stop()
        """
    )
    return capture(
        [str(python_bin), "-c", probe],
        env=runtime_env,
        check=False,
    ).returncode == 0


def ensure_venv(harness_dir: pathlib.Path, python_bin: str, skip_install: bool) -> pathlib.Path:
    venv_dir = harness_dir / ".venv-walite"
    py = venv_python(venv_dir)
    if not py.exists():
        log(f"Creating virtualenv at {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir))

        current = capture([str(py), "-c", "import sys; print(sys.version_info[:2])"])
        if "(3, 10)" not in current.stdout and "(3, 11)" not in current.stdout:
            shutil.rmtree(str(venv_dir), ignore_errors=True)
            run([python_bin, "-m", "venv", str(venv_dir)])

    runtime_env = dict(os.environ)
    runtime_env.update(
        {
            "PIP_CACHE_DIR": str(harness_dir / ".cache" / "pip"),
            "PLAYWRIGHT_BROWSERS_PATH": str(harness_dir / ".playwright-browsers"),
            "TMPDIR": str(harness_dir / ".tmp"),
        }
    )
    for path in (
        runtime_env["PIP_CACHE_DIR"],
        runtime_env["PLAYWRIGHT_BROWSERS_PATH"],
        runtime_env["TMPDIR"],
    ):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    dependencies_ready = runtime_dependencies_ready(py, runtime_env)
    if skip_install:
        log("Skipping runtime installation by request")
    elif dependencies_ready:
        log("Runtime dependencies and Chromium are already ready")
    else:
        run(
            [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            env=runtime_env,
        )
        run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--use-deprecated=legacy-resolver",
                "-r",
                "requirements.txt",
            ],
            cwd=harness_dir,
            env=runtime_env,
        )
        run([str(py), "-m", "pip", "install", *RUNTIME_REQUIREMENTS], env=runtime_env)
        run(
            [
                str(py),
                "-m",
                "nltk.downloader",
                "-d",
                str(venv_dir / "nltk_data"),
                "punkt",
            ],
            env=runtime_env,
        )
        run(
            [str(py), "-m", "playwright", "install", "chromium"],
            cwd=harness_dir,
            env=runtime_env,
        )
        run(
            [str(py), "-m", "pip", "install", "-e", "."],
            cwd=harness_dir,
            env=runtime_env,
        )

    patch_webrl_action_parser(harness_dir)
    patch_auto_login_timeout(harness_dir)
    return py


def env_with_venv_python(env: Mapping[str, str], python_bin: pathlib.Path) -> dict[str, str]:
    updated = dict(env)
    bin_dir = str(python_bin.parent)
    old_path = updated.get("PATH", "")
    updated["PATH"] = bin_dir if not old_path else f"{bin_dir}{os.pathsep}{old_path}"
    updated["NLTK_DATA"] = str(python_bin.parent.parent / "nltk_data")
    harness_dir = python_bin.parent.parent.parent
    updated["PLAYWRIGHT_BROWSERS_PATH"] = str(harness_dir / ".playwright-browsers")
    updated["PIP_CACHE_DIR"] = str(harness_dir / ".cache" / "pip")
    updated["TMPDIR"] = str(harness_dir / ".tmp")
    return updated


def env_with_harness_pythonpath(
    env: Mapping[str, str], harness_dir: pathlib.Path
) -> dict[str, str]:
    updated = dict(env)
    harness_path = str(harness_dir)
    old_path = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = (
        harness_path if not old_path else f"{harness_path}{os.pathsep}{old_path}"
    )
    return updated


def build_run_command(python_bin: str, config: RunConfig) -> list[str]:
    return [
        python_bin,
        "run.py",
        "--instruction_path",
        "agent/prompts/jsons/p_webrl_chat.json",
        "--test_start_idx",
        str(config.test_start_idx),
        "--test_end_idx",
        str(config.test_end_idx),
        "--result_dir",
        str(config.result_dir),
        "--test_config_base_dir",
        "config_files/wa/test_webarena_lite",
        "--provider",
        "openai",
        "--model",
        config.model,
        "--mode",
        "chat",
        "--planner_ip",
        "",
        "--temperature",
        str(config.temperature),
        "--top_p",
        str(config.top_p),
        "--max_obs_length",
        "0",
        "--max_tokens",
        str(config.max_tokens),
        "--max_steps",
        str(config.max_steps),
        "--viewport_width",
        str(config.viewport_width),
        "--viewport_height",
        str(config.viewport_height),
        "--parsing_failure_th",
        str(config.parsing_failure_th),
        "--repeating_action_failure_th",
        str(config.repeating_action_failure_th),
        "--action_set_tag",
        "webrl_id",
        "--observation_type",
        "webrl",
    ]


def task_config_fingerprint(env: Mapping[str, str]) -> str:
    keys = ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE")
    payload = json.dumps({key: env.get(key, "") for key in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_test_data(
    harness_dir: pathlib.Path, python_bin: pathlib.Path, env: Mapping[str, str]
) -> None:
    out_dir = harness_dir / "config_files" / "wa" / "test_webarena_lite"
    fingerprint_path = out_dir / ".site_fingerprint"
    expected_fingerprint = task_config_fingerprint(env)
    existing = list(out_dir.glob("*.json")) if out_dir.exists() else []
    existing_fingerprint = (
        fingerprint_path.read_text(encoding="utf-8").strip()
        if fingerprint_path.exists()
        else ""
    )
    if len(existing) == TASK_COUNT and existing_fingerprint == expected_fingerprint:
        log("Task configs already exist")
        return
    if out_dir.exists():
        for path in out_dir.glob("*.json"):
            path.unlink()
    run([str(python_bin), "scripts/generate_test_data.py"], cwd=harness_dir, env=env)
    generated = list(out_dir.glob("*.json"))
    if len(generated) != TASK_COUNT:
        die(f"Expected {TASK_COUNT} task configs, found {len(generated)} in {out_dir}")
    fingerprint_path.write_text(expected_fingerprint + "\n", encoding="utf-8")


def prepare_login(harness_dir: pathlib.Path, env: Mapping[str, str], skip: bool) -> None:
    if skip:
        log("Skipping prepare.sh by request")
        return

    auth_dir = harness_dir / ".auth"
    refresh_dir = harness_dir / ".auth.refresh"
    shutil.rmtree(refresh_dir, ignore_errors=True)
    refresh_dir.mkdir(parents=True)

    for combination in REQUIRED_AUTH_COMBINATIONS:
        log(f"Refreshing login state: {'.'.join(combination)}")
        run(
            [
                "python",
                "browser_env/auto_login.py",
                "--auth_folder",
                str(refresh_dir),
                "--site_list",
                *combination,
            ],
            cwd=harness_dir,
            env=env,
        )

    expected = {
        f"{'.'.join(combination)}_state.json"
        for combination in REQUIRED_AUTH_COMBINATIONS
    }
    generated = {path.name for path in refresh_dir.glob("*_state.json")}
    if generated != expected:
        die(
            "Login state set mismatch: "
            f"expected {sorted(expected)}, found {sorted(generated)}"
        )

    validator = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        from browser_env.auto_login import (
            EXACT_MATCH,
            KEYWORDS,
            SITES,
            URLS,
            get_site_comb_from_filepath,
            is_expired,
        )

        for state_path in sorted(Path(sys.argv[1]).glob("*_state.json")):
            for site in get_site_comb_from_filepath(str(state_path)):
                index = SITES.index(site)
                if is_expired(
                    state_path,
                    URLS[index],
                    KEYWORDS[index],
                    EXACT_MATCH[index],
                ):
                    raise SystemExit(f"expired login state: {state_path.name} ({site})")
        """
    )
    run(["python", "-c", validator, str(refresh_dir)], cwd=harness_dir, env=env)

    if auth_dir.exists():
        backup = harness_dir / f".auth.before-{time.time_ns()}"
        auth_dir.rename(backup)
        log(f"Previous login state saved at {backup}")
    refresh_dir.rename(auth_dir)
    log("Fresh login states validated and installed")


def http_status(url: str, timeout: float) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "walite-runner/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return True, str(response.status)
    except urllib.error.HTTPError as exc:
        return 200 <= exc.code < 400, str(exc.code)
    except Exception as exc:
        return False, exc.__class__.__name__


def check_sites(env: Mapping[str, str], timeout: float = 5.0) -> list[tuple[str, str, bool, str]]:
    checks = [
        ("SHOPPING", env["SHOPPING"]),
        ("SHOPPING_ADMIN", env["SHOPPING_ADMIN"]),
        ("REDDIT", env["REDDIT"]),
        ("GITLAB", env["GITLAB"]),
        ("MAP", env["MAP"]),
        ("MAP_TILE", env["MAP_TILE"]),
        ("WIKIPEDIA", env["WIKIPEDIA"]),
        ("HOMEPAGE", env["HOMEPAGE"]),
    ]
    results = []
    for name, url in checks:
        ok, status = http_status(url, timeout)
        results.append((name, url, ok, status))
    return results


def require_sites(env: Mapping[str, str], allow_unhealthy: bool) -> None:
    results = check_sites(env)
    failed = []
    for name, url, ok, status in results:
        marker = "ok" if ok else "FAIL"
        log(f"site {name:<14} {marker:<4} {status:<18} {url}")
        if not ok:
            failed.append((name, url, status))
    if failed and not allow_unhealthy:
        detail = "\n".join(f"  - {name}: {status} {url}" for name, url, status in failed)
        die(
            "WebArena-Lite websites are not all reachable. "
            "Start the benchmark websites or override their URLs, then rerun.\n" + detail
        )


def score_result(
    harness_dir: pathlib.Path,
    python_bin: pathlib.Path,
    result_dir: pathlib.Path,
    env: Mapping[str, str],
) -> pathlib.Path:
    score_path = result_dir / "score.txt"
    proc = capture(
        [str(python_bin), "score.py", str(result_dir)], cwd=harness_dir, env=env, check=False
    )
    score_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        die(f"score.py failed; see {score_path}")
    return score_path


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def action_files(result_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    return sorted((result_dir / "actions").glob("*.json"))


def validate_results(result_dir: pathlib.Path, expected_task_ids: Iterable[int]) -> None:
    expected = set(expected_task_ids)
    seen: dict[int, float] = {}
    problems = []
    for path in action_files(result_dir):
        try:
            item = load_json(path)
            task_id = int(item.get("task_id", path.stem))
            score = float(item["score"])
        except Exception as exc:
            problems.append(f"{path.name}: unreadable result ({exc.__class__.__name__})")
            continue
        if task_id in seen:
            problems.append(f"task {task_id}: duplicate action result")
        seen[task_id] = score
        if score < 0:
            problems.append(f"task {task_id}: invalid environment score {score}")

    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)
    if missing:
        problems.append(f"missing task results: {missing}")
    if unexpected:
        problems.append(f"unexpected task results: {unexpected}")
    if problems:
        die("Result set is incomplete or invalid:\n  - " + "\n  - ".join(problems))


def report_site_name(sites) -> str:
    if isinstance(sites, str):
        sites = [sites]
    names = {
        ("shopping",): "shopping",
        ("shopping_admin",): "shopping_admin",
        ("reddit",): "reddit",
        ("gitlab",): "gitlab",
        ("map",): "map",
    }
    return names.get(tuple(sites), "+".join(sites))


def write_report(
    harness_dir: pathlib.Path,
    result_dir: pathlib.Path,
    model: str,
    score_path: pathlib.Path,
    expected_task_ids: Iterable[int] = range(TASK_COUNT),
) -> pathlib.Path:
    expected_ids = list(expected_task_ids)
    expected_set = set(expected_ids)
    total_expected = len(expected_ids)
    config_raw = harness_dir / "config_files" / "wa" / "test_webarena_lite.raw.json"
    raw_configs = load_json(config_raw) if config_raw.exists() else []
    by_task = {int(item["task_id"]): item for item in raw_configs}

    scores = {}
    step_counts = {}
    for path in action_files(result_dir):
        try:
            item = load_json(path)
        except Exception:
            continue
        task_id = int(item.get("task_id", path.stem))
        if task_id not in expected_set:
            continue
        score = float(item.get("score", -1))
        if score >= 0:
            scores[task_id] = score
        step_counts[task_id] = len(item.get("actions", []))

    success = sum(1 for score in scores.values() if score >= 1)
    finished = len(scores)
    per_site = {}
    for task_id in expected_ids:
        raw = by_task.get(task_id, {})
        site = report_site_name(raw.get("sites", ["unknown"]))
        per_site.setdefault(site, []).append(scores.get(task_id, 0))

    lines = [
        f"# {model} WebArena-Lite Evaluation Report",
        "",
        f"- Generated at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Result directory: `{result_dir}`",
        "- Benchmark: `VAB-WebArena-Lite`",
        "- Observation: `webrl` simplified HTML text",
        "- Action space: `webrl_id`",
        "- Prompt: `agent/prompts/jsons/p_webrl_chat.json`",
        f"- Finished tasks: {finished}/{total_expected}",
        f"- Successful tasks: {success}/{total_expected}",
        f"- Overall SR: **{success / total_expected * 100:.2f}%**",
        "",
        "## Per-site SR",
        "",
        "| Site | Success | Total | SR |",
        "|---|---:|---:|---:|",
    ]
    for site, values in sorted(per_site.items()):
        ok = sum(1 for value in values if value >= 1)
        total = len(values)
        lines.append(f"| {site} | {ok} | {total} | {ok / total * 100:.2f}% |")

    lines.extend(["", "## Step Stats", ""])
    if step_counts:
        values = list(step_counts.values())
        lines.append(f"- Average recorded actions: {sum(values) / len(values):.2f}")
        lines.append(f"- Max recorded actions: {max(values)}")
    else:
        lines.append("- No action files found.")

    lines.extend(["", "## Official score.py Output", "", "```text"])
    if score_path.exists():
        lines.append(score_path.read_text(encoding="utf-8").strip())
    lines.extend(["```", ""])

    path = result_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run VAB-WebArena-Lite with an OpenAI-compatible chat model.",
        epilog=textwrap.dedent(
            """
            Minimal usage:
              export OPENAI_API_KEY=...
              export OPENAI_BASE_URL=https://litellm.local.lexmount.net/v1
              export OPENAI_MODEL=glm-5.2
              python run_webarena_lite.py

            Useful overrides:
              python run_webarena_lite.py --smoke
              python run_webarena_lite.py --server webarena-hostname
              SHOPPING=http://host:7770 REDDIT=http://host:9999 \
                python run_webarena_lite.py
            """
        ),
    )
    parser.add_argument("--work-dir", type=pathlib.Path, default=None)
    parser.add_argument("--result-dir", type=pathlib.Path, default=None)
    parser.add_argument("--runtime-root", type=pathlib.Path, default=None)
    parser.add_argument("--server", default=None)
    parser.add_argument("--map-server", default=os.environ.get("WEBARENA_MAP_SERVER"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=TASK_COUNT)
    parser.add_argument("--smoke", action="store_true", help="Run only task 0")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--allow-unhealthy-sites", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_root = pathlib.Path(__file__).resolve().parent
    runtime_root = args.runtime_root or default_runtime_root(script_root)
    work_dir = args.work_dir or runtime_root / ".webarena_lite"
    process_env = dict(os.environ)
    if args.server:
        process_env["WEBARENA_SERVER"] = args.server
    server = args.server or process_env.get("WEBARENA_SERVER", DEFAULT_SERVER)
    user_env = environ_with_site_env(runtime_root, process_env)

    model = require_env(user_env, "OPENAI_MODEL")
    result_dir = (args.result_dir or default_result_dir(runtime_root, model)).resolve()
    if args.smoke:
        args.start, args.end = 0, 1
        if args.result_dir is None:
            result_dir = runtime_root / "results" / f"{slug_model(model)}_webarena_lite_smoke"

    if not 0 <= args.start < args.end <= TASK_COUNT:
        die(f"Task range must satisfy 0 <= start < end <= {TASK_COUNT}")
    expected_task_ids = list(range(args.start, args.end))

    env = build_harness_env(user_env, server=server, map_server=args.map_server)
    log(f"Using model: {model}")
    log(f"Using work dir: {work_dir}")
    log(f"Using result dir: {result_dir}")

    if not args.score_only:
        require_sites(env, allow_unhealthy=args.allow_unhealthy_sites)

    if args.skip_bootstrap:
        harness_dir = work_dir / "VisualAgentBench" / "VAB-WebArena-Lite"
        if not (harness_dir / "run.py").exists():
            die(f"--skip-bootstrap was used but harness is missing at {harness_dir}")
    else:
        harness_dir = ensure_checkout(work_dir)

    python_bin = pathlib.Path(find_python310())
    if not args.skip_bootstrap:
        python_bin = ensure_venv(harness_dir, str(python_bin), skip_install=args.skip_install)
        env = env_with_venv_python(env, python_bin)
    env = env_with_harness_pythonpath(env, harness_dir)

    result_dir.mkdir(parents=True, exist_ok=True)

    generate_test_data(harness_dir, python_bin, env)
    prepare_login(harness_dir, env, skip=args.skip_prepare or args.score_only)

    if not args.score_only:
        config = RunConfig(
            result_dir=result_dir,
            test_start_idx=args.start,
            test_end_idx=args.end,
            model=model,
        )
        command = build_run_command(str(python_bin), config)
        started_at = time.time()
        run(command, cwd=harness_dir, env=env)
        log(f"Evaluation command finished in {(time.time() - started_at) / 60:.1f} minutes")

    validate_results(result_dir, expected_task_ids)
    score_path = score_result(harness_dir, python_bin, result_dir, env)
    report_path = write_report(
        harness_dir,
        result_dir,
        model,
        score_path,
        expected_task_ids=expected_task_ids,
    )
    log(f"Score written to {score_path}")
    log(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
