"""Unified LexBench command interface for frozen GPT-5.5 and Qwen3-8B experiments."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from .qwen3_8b import campaign as stress_campaign
from .qwen3_8b import monitor, stress_stage2
from .qwen3_8b.official import (
    build_quality_command,
    ensure_checkout,
    freeze_dependencies,
    resolve_output_marker,
    sync_official_config,
    validate_environment,
)
from .qwen3_8b.protocol import PROTOCOL, resolve_runtime_paths
from .qwen3_8b.report import generate_campaign_report

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
QWEN_CONFIG = PROJECT_ROOT / "experiments" / "qwen3-8b-lexbench" / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    qwen = subparsers.add_parser("qwen3-8b", help="Run the frozen Qwen3-8B protocol")
    qwen.add_argument("--env-file", type=pathlib.Path, required=True)
    qwen.add_argument("--runtime-root", type=pathlib.Path, default=pathlib.Path("/data/wf/sxh"))
    qwen.add_argument("--backend", choices=("lexmount", "local", "all"), default="all")
    qwen.add_argument("--mode", choices=("quality", "stress", "all"), default="all")
    qwen.add_argument(
        "--stage",
        choices=("prepare", "rollout", "judge", "report", "all"),
        default="all",
    )
    qwen.add_argument("--campaign-id")
    qwen.add_argument(
        "--task-count",
        type=int,
        default=PROTOCOL.quality_task_count,
        help="Official All task count (210); a smaller deterministic first_n slice is smoke-only.",
    )
    qwen.add_argument("--resume", action="store_true")
    qwen.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def selected_backends(value: str) -> tuple[str, ...]:
    return ("lexmount", "local") if value == "all" else (value,)


def prepare_qwen(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    if not args.env_file.is_file():
        raise RuntimeError(f"environment file is not readable: {args.env_file}")
    load_dotenv(args.env_file, override=False)
    paths = resolve_runtime_paths(PROJECT_ROOT, args.runtime_root.resolve())
    ensure_checkout(paths.checkout)
    sync_official_config(paths.checkout, QWEN_CONFIG)
    validate_environment(
        paths.runtime_root, paths.results_root, selected_backends(args.backend), dict(os.environ)
    )
    return paths.checkout, paths.results_root


def quality_concurrency(task_count: int) -> int:
    return min(PROTOCOL.quality_concurrency, task_count)


def quality_cell_root(
    results_root: pathlib.Path, campaign_id: str, backend: str, task_count: int
) -> pathlib.Path:
    name = f"quality_{backend}_c{quality_concurrency(task_count)}"
    if task_count != PROTOCOL.quality_task_count:
        name += f"_n{task_count}"
    return results_root / campaign_id / name


def quality_command(
    *,
    checkout: pathlib.Path,
    results_root: pathlib.Path,
    campaign_id: str,
    backend: str,
    task_count: int,
    rollout_only: bool,
    resume: bool,
) -> tuple[list[str], pathlib.Path, pathlib.Path | None]:
    cell = quality_cell_root(results_root, campaign_id, backend, task_count)
    marker = cell / "official_run_dir.txt"
    existing: pathlib.Path | None = None
    command = build_quality_command(
        backend,
        marker,
        task_count=task_count,
        concurrency=quality_concurrency(task_count),
    )
    if rollout_only:
        command.append("--skip-eval")
    if resume:
        if not marker.is_file():
            raise RuntimeError(f"resume marker is missing: {marker}")
        existing = resolve_output_marker(checkout, marker)
        command.extend(("--timestamp", existing.name, "--skip-completed"))
    elif marker.exists():
        raise RuntimeError(f"quality cell already exists; pass --resume: {cell}")
    return command, marker, existing


def run_quality(
    args: argparse.Namespace, checkout: pathlib.Path, results_root: pathlib.Path
) -> None:
    rollout_only = args.stage == "rollout"
    experiment_root = (
        checkout
        / "experiments"
        / "LexBench-Browser"
        / "All"
        / "browser-use"
        / PROTOCOL.agent_model_id
    )
    for backend in selected_backends(args.backend):
        command, marker, existing = quality_command(
            checkout=checkout,
            results_root=results_root,
            campaign_id=args.campaign_id,
            backend=backend,
            task_count=args.task_count,
            rollout_only=rollout_only,
            resume=args.resume,
        )
        if args.dry_run:
            print(shlex.join(command))
            continue
        cell = marker.parent
        cell.mkdir(parents=True, exist_ok=True)
        monitor_args = [
            "--output-dir",
            str(cell / "monitor"),
            "--cwd",
            str(checkout),
            "--experiment-root",
            str(experiment_root),
            "--expected-tasks",
            str(args.task_count),
            "--baseline-seconds",
            str(PROTOCOL.baseline_seconds),
            "--interval-seconds",
            str(PROTOCOL.sample_interval_seconds),
        ]
        if not rollout_only:
            monitor_args.append("--has-judge")
        if existing is not None:
            monitor_args.extend(("--existing-run-dir", str(existing)))
        status = monitor.main([*monitor_args, "--", *command])
        if status != 0:
            raise RuntimeError(f"official quality run failed for {backend}: exit {status}")


def run_stress_rollout(args: argparse.Namespace) -> pathlib.Path:
    stress_campaign.main(
        [
            "campaign",
            "--campaign-id",
            args.campaign_id,
            "--runtime-root",
            str(args.runtime_root),
            "--env-file",
            str(args.env_file),
            "--backend",
            args.backend,
        ]
    )
    return (
        args.runtime_root / "results" / "lexbench" / args.campaign_id / "stress_process_attributed"
    )


def run_stress_judge(args: argparse.Namespace, checkout: pathlib.Path) -> pathlib.Path:
    stress_root = (
        args.runtime_root / "results" / "lexbench" / args.campaign_id / "stress_process_attributed"
    )
    rollout_summary = stress_root / "rollout_summary.json"
    if not rollout_summary.is_file():
        raise RuntimeError(f"stress rollout summary is missing: {rollout_summary}")
    stage_dir = stress_root / "stage2_gpt54_c5"
    stress_stage2.main(
        [
            "run",
            "--checkout",
            str(checkout),
            "--rollout-summary",
            str(rollout_summary),
            "--stage-dir",
            str(stage_dir),
            "--workers",
            "5",
        ]
    )
    return stage_dir


def dry_run(args: argparse.Namespace) -> int:
    checkout = args.runtime_root / ".lexbench" / "browseruse-agent-bench"
    results_root = args.runtime_root / "results" / "lexbench"
    if args.mode in {"quality", "all"} and args.stage in {"rollout", "all"}:
        for backend in selected_backends(args.backend):
            command, _, _ = quality_command(
                checkout=checkout,
                results_root=results_root,
                campaign_id=args.campaign_id,
                backend=backend,
                task_count=args.task_count,
                rollout_only=args.stage == "rollout",
                resume=args.resume,
            )
            print(shlex.join(command))
    if args.mode in {"stress", "all"} and args.stage in {"rollout", "judge", "all"}:
        print(
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "lexbrowser_eval.lexbench.qwen3_8b.campaign",
                    "campaign",
                    "--campaign-id",
                    args.campaign_id,
                    "--runtime-root",
                    str(args.runtime_root),
                    "--env-file",
                    str(args.env_file),
                    "--backend",
                    args.backend,
                ]
            )
        )
    return 0


def run_qwen(args: argparse.Namespace) -> int:
    if not 1 <= args.task_count <= PROTOCOL.quality_task_count:
        raise RuntimeError(f"--task-count must be in 1..{PROTOCOL.quality_task_count}")
    if not args.campaign_id:
        args.campaign_id = pathlib.Path(os.environ.get("LEXBENCH_CAMPAIGN_ID", "")).name
    if not args.campaign_id:
        from datetime import UTC, datetime

        args.campaign_id = f"lexbench_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    args.runtime_root = args.runtime_root.resolve()
    if args.dry_run:
        return dry_run(args)
    if args.stage == "judge" and args.mode in {"quality", "all"}:
        raise RuntimeError(
            "standalone quality Judge is not supported; use --stage all or official run-eval"
        )

    checkout, results_root = prepare_qwen(args)
    if args.stage == "prepare":
        print(checkout)
        return 0
    if args.mode in {"quality", "all"} and args.stage in {"rollout", "all"}:
        run_quality(args, checkout, results_root)
    if args.mode in {"stress", "all"} and args.stage in {"rollout", "all"}:
        run_stress_rollout(args)
    if args.mode in {"stress", "all"} and args.stage in {"judge", "all"}:
        run_stress_judge(args, checkout)
    result_root = results_root / args.campaign_id
    if args.stage in {"report", "all"}:
        json_report, markdown_report = generate_campaign_report(
            checkout=checkout,
            results_root=results_root,
            campaign_id=args.campaign_id,
            backends=selected_backends(args.backend),
            include_quality=args.mode in {"quality", "all"},
            include_stress=args.mode in {"stress", "all"},
            quality_task_count=args.task_count,
        )
        print(json_report)
        print(markdown_report)
    print(result_root)
    if (checkout / ".venv/bin/python").exists() and (
        checkout / ".venvs/browser_use/bin/python"
    ).exists():
        freeze_dependencies(checkout, result_root / "dependency-freeze.txt")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.experiment == "qwen3-8b":
        return run_qwen(args)
    raise AssertionError(args.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
