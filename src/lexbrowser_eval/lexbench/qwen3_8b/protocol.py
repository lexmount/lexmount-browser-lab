from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    upstream_repo: str = "https://github.com/lexmount/browseruse-agent-bench.git"
    upstream_commit: str = "ccd5fcbdfb975257b2ce38161dc9bc2ab294b420"
    agent_model_id: str = "qwen3_8B"
    model_config_name: str = "qwen3-8B"
    quality_split: str = "All"
    quality_task_count: int = 210
    quality_sha256: str = "90fc09b2fbdcd391d70924d1fee069534784bc133aaefabf26d3892e48983108"
    quality_concurrency: int = 10
    judge_model: str = "gpt-5.4"
    judge_strategy: str = "stepwise"
    sample_interval_seconds: float = 1.0
    baseline_seconds: int = 30


@dataclass(frozen=True)
class RuntimePaths:
    project_root: pathlib.Path
    runtime_root: pathlib.Path
    checkout: pathlib.Path
    results_root: pathlib.Path


PROTOCOL = Protocol()
STRESS_SCHEDULE = (
    (20, 20, "lexmount"),
    (20, 20, "local"),
    (50, 50, "local"),
    (50, 50, "lexmount"),
)
BACKENDS = ("lexmount", "local")
REQUIRED_ENV = (
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    "QWEN_MODEL_ID",
    "LEXBENCH_JUDGE_API_KEY",
    "LEXBENCH_JUDGE_BASE_URL",
    "LEXBENCH_JUDGE_MODEL",
    "LEXMOUNT_API_KEY",
    "LEXMOUNT_PROJECT_ID",
)


def resolve_runtime_paths(project_root: pathlib.Path, runtime_root: pathlib.Path) -> RuntimePaths:
    return RuntimePaths(
        project_root=project_root,
        runtime_root=runtime_root,
        checkout=runtime_root / ".lexbench" / "browseruse-agent-bench",
        results_root=runtime_root / "results" / "lexbench",
    )


def required_env_names(backends: tuple[str, ...]) -> tuple[str, ...]:
    names = list(REQUIRED_ENV[:6])
    if "lexmount" in backends:
        names.extend(REQUIRED_ENV[6:])
    return tuple(names)
