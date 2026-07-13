from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile

from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version

from .protocol import PROTOCOL, STRESS_SCHEDULE, required_env_names


def capture_text(command: list[str], cwd: pathlib.Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkout(checkout: pathlib.Path) -> None:
    if checkout.exists():
        validate_checkout(checkout)
        return
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", PROTOCOL.upstream_repo, str(checkout)], check=True)
    subprocess.run(
        ["git", "checkout", "--detach", PROTOCOL.upstream_commit],
        cwd=checkout,
        check=True,
    )
    validate_checkout(checkout)


def validate_checkout(checkout: pathlib.Path) -> None:
    head = capture_text(["git", "rev-parse", "HEAD"], checkout).strip()
    if head != PROTOCOL.upstream_commit:
        raise RuntimeError(f"Official checkout commit mismatch: {head}")
    status = capture_text(["git", "status", "--porcelain"], checkout).rstrip()
    unexpected = [line for line in status.splitlines() if line[3:] != "config.yaml"]
    if unexpected:
        raise RuntimeError(f"Official checkout worktree is not clean: {'; '.join(unexpected)}")
    data = checkout / "browseruse_bench" / "data" / "LexBench-Browser"
    path = data / "task.jsonl"
    actual = sha256_file(path)
    if actual != PROTOCOL.quality_sha256:
        raise RuntimeError(f"Dataset hash mismatch for {path.name}: {actual}")


def validate_environment(
    runtime_root: pathlib.Path,
    results_root: pathlib.Path,
    backends: tuple[str, ...],
    environ: dict[str, str],
) -> dict[str, object]:
    missing: list[str] = []
    if platform.system() != "Linux":
        missing.append("platform=Linux")
    if runtime_root != pathlib.Path("/data/wf/sxh"):
        missing.append("runtime_root=/data/wf/sxh")
    tools = {name: shutil.which(name) for name in ("git", "uv", "xvfb-run", "nvidia-smi")}
    missing.extend(name for name, path in tools.items() if not path)
    missing.extend(
        name for name in required_env_names(backends) if not environ.get(name, "").strip()
    )
    required_values = {
        "QWEN_MODEL_ID": PROTOCOL.agent_model_id,
        "LEXBENCH_JUDGE_MODEL": PROTOCOL.judge_model,
    }
    for name, expected in required_values.items():
        actual = environ.get(name, "").strip()
        if actual and actual != expected:
            missing.append(f"{name}={expected}")
    results_root.mkdir(parents=True, exist_ok=True)
    if not os.access(results_root, os.W_OK):
        missing.append(f"writable:{results_root}")
    if missing:
        raise RuntimeError("LexBench preflight failed: " + ", ".join(missing))
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git": tools["git"],
        "uv": tools["uv"],
        "xvfb_run": tools["xvfb-run"],
        "nvidia_smi": tools["nvidia-smi"],
        "runtime_root": str(runtime_root),
        "configured_env_names": sorted(required_env_names(backends)),
    }


_DISTRIBUTION_SCRIPT = (
    "import importlib.metadata as m, json\n"
    "records = [[d.metadata.get('Name'), d.version] for d in m.distributions() "
    "if d.metadata.get('Name') and d.version]\n"
    "print(json.dumps(records, ensure_ascii=True))\n"
)


def _normalize_distribution_payload(payload: str) -> list[str]:
    try:
        records = json.loads(payload)
    except json.JSONDecodeError:
        raise RuntimeError("Invalid dependency metadata payload") from None
    if not isinstance(records, list):
        raise RuntimeError("Invalid dependency metadata payload")
    packages: dict[str, str] = {}
    for record in records:
        if (
            not isinstance(record, list)
            or len(record) != 2
            or not all(isinstance(item, str) and item.strip() for item in record)
        ):
            raise RuntimeError("Invalid dependency metadata record")
        raw_name, raw_version = record
        try:
            name = canonicalize_name(raw_name, validate=True)
        except InvalidName:
            raise RuntimeError("Invalid dependency name metadata") from None
        try:
            version = str(Version(raw_version))
        except InvalidVersion:
            raise RuntimeError("Invalid dependency version metadata") from None
        previous = packages.get(name)
        if previous is not None and previous != version:
            raise RuntimeError(f"Conflicting installed versions for {name}")
        packages[name] = version
    if not packages:
        raise RuntimeError("No dependency metadata records")
    return [f"{name}=={packages[name]}" for name in sorted(packages)]


def freeze_dependencies(checkout: pathlib.Path, output: pathlib.Path) -> None:
    candidates = (
        ("official-root", checkout / ".venv" / "bin" / "python"),
        ("browser-use", checkout / ".venvs" / "browser_use" / "bin" / "python"),
    )
    missing = [name for name, python in candidates if not python.exists()]
    if missing:
        raise RuntimeError("Missing official Python environments: " + ", ".join(missing))
    sections: list[str] = []
    for name, python in candidates:
        payload = subprocess.run(
            [str(python), "-c", _DISTRIBUTION_SCRIPT],
            cwd=checkout,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        packages = _normalize_distribution_payload(payload)
        sections.extend((f"[{name}]", *packages, ""))

    content = "\n".join(sections).rstrip() + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _official_base(backend: str, marker: pathlib.Path) -> list[str]:
    command = [
        "uv",
        "run",
        "bubench",
        "run-eval",
        "--agent",
        "browser-use",
        "--data",
        "LexBench-Browser",
        "--model",
        PROTOCOL.model_config_name,
        "--browser-id",
        backend,
        "--report-output-dir",
        str(marker),
    ]
    if backend == "local":
        return ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", *command]
    return command


def build_quality_command(
    backend: str,
    marker: pathlib.Path,
    *,
    task_count: int = PROTOCOL.quality_task_count,
    concurrency: int = PROTOCOL.quality_concurrency,
) -> list[str]:
    """Build the upstream LexBench quality command.

    A smaller deterministic ``first_n`` slice is permitted only for isolated
    smoke validation.  The default remains the official All/210 protocol.
    """
    if not 1 <= task_count <= PROTOCOL.quality_task_count:
        raise ValueError(f"task_count must be in 1..{PROTOCOL.quality_task_count}")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    command = _official_base(backend, marker)
    command.extend(["--split", PROTOCOL.quality_split])
    if task_count == PROTOCOL.quality_task_count:
        command.extend(["--mode", "all"])
    else:
        command.extend(["--mode", "first_n", "--count", str(task_count), "--no-group-by-site"])
    command.extend(
        ["--concurrency", str(concurrency), "--eval-strategy", PROTOCOL.judge_strategy]
    )
    return command


def build_stress_command(
    backend: str,
    task_count: int,
    concurrency: int,
    marker: pathlib.Path,
) -> list[str]:
    allowed = {(count, level, backend_id) for count, level, backend_id in STRESS_SCHEDULE}
    if (task_count, concurrency, backend) not in allowed:
        raise ValueError("Stress cell is not part of the frozen protocol")
    command = _official_base(backend, marker)
    command.extend(
        [
            "--split",
            "All",
            "--mode",
            "first_n",
            "--count",
            str(task_count),
            "--no-group-by-site",
            "--concurrency",
            str(concurrency),
            "--skip-eval",
        ]
    )
    return command


def sync_official_config(checkout: pathlib.Path, source: pathlib.Path) -> pathlib.Path:
    target = checkout / "config.yaml"
    shutil.copyfile(source, target)
    return target


def resolve_output_marker(checkout: pathlib.Path, marker: pathlib.Path) -> pathlib.Path:
    raw = marker.read_text(encoding="utf-8").strip()
    path = pathlib.Path(raw)
    return path if path.is_absolute() else checkout / path
