#!/usr/bin/env python3
"""Export NeMo RL TensorBoard scalars and render the final training report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_latest_run(log_root: Path) -> tuple[Path, EventAccumulator]:
    candidates: list[tuple[int, float, Path, EventAccumulator]] = []
    for run_dir in log_root.glob("exp_*"):
        if not any(run_dir.rglob("events.out.tfevents.*")):
            continue
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        point_count = sum(len(accumulator.Scalars(tag)) for tag in accumulator.Tags()["scalars"])
        candidates.append((point_count, run_dir.stat().st_mtime, run_dir, accumulator))
    if not candidates:
        raise SystemExit(f"No TensorBoard event run found under {log_root}")
    _, _, run_dir, accumulator = max(candidates)
    return run_dir, accumulator


def select_tags(tags: list[str], keywords: tuple[str, ...]) -> list[str]:
    return [tag for tag in tags if any(word in tag.lower() for word in keywords)]


def plot_group(
    accumulator: EventAccumulator,
    tags: list[str],
    path: Path,
    title: str,
    ylabel: str,
) -> bool:
    usable = [tag for tag in tags if len(accumulator.Scalars(tag)) >= 1]
    if not usable:
        return False
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=160)
    for tag in usable[:12]:
        events = accumulator.Scalars(tag)
        axis.plot([event.step for event in events], [event.value for event in events], label=tag)
    axis.set_title(title)
    axis.set_xlabel("Training step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return True


def linear_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def fmt(value: float | None) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.6g}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir, accumulator = load_latest_run(args.log_root)
    tags = sorted(accumulator.Tags()["scalars"])

    with (args.output_dir / "training_scalars.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "step", "wall_time", "value"])
        for tag in tags:
            for event in accumulator.Scalars(tag):
                writer.writerow([tag, event.step, event.wall_time, event.value])

    reward_tags = select_tags(tags, ("reward",))
    behavior_tags = select_tags(
        tags,
        (
            "assistant_turns",
            "tool_calls",
            "act_calls",
            "observe_calls",
            "extract_calls",
            "navigate_calls",
            "generation_tokens",
            "response_length",
        ),
    )
    optimization_tags = select_tags(
        tags, ("learning_rate", "lr", "loss", "grad_norm", "kl")
    )

    plots: list[tuple[str, str]] = []
    if plot_group(
        accumulator,
        reward_tags,
        args.output_dir / "reward_vs_step.png",
        "Reward vs. training step",
        "Reward",
    ):
        plots.append(("Reward", "reward_vs_step.png"))
    if plot_group(
        accumulator,
        behavior_tags,
        args.output_dir / "browser_behavior_vs_step.png",
        "Browser rollout behavior vs. training step",
        "Per-rollout / aggregate metric",
    ):
        plots.append(("推理步数与工具调用", "browser_behavior_vs_step.png"))
    if plot_group(
        accumulator,
        optimization_tags,
        args.output_dir / "optimization_vs_step.png",
        "Optimization metrics vs. training step",
        "Metric value",
    ):
        plots.append(("学习率、loss 与优化指标", "optimization_vs_step.png"))

    primary_reward = min(
        reward_tags,
        key=lambda tag: (
            "mean" not in tag.lower(),
            "train" not in tag.lower(),
            len(tag),
        ),
    ) if reward_tags else None
    reward_points = (
        [(event.step, event.value) for event in accumulator.Scalars(primary_reward)]
        if primary_reward
        else []
    )
    window = min(10, max(1, len(reward_points) // 4)) if reward_points else 0
    first_mean = (
        sum(value for _, value in reward_points[:window]) / window if window else None
    )
    last_mean = (
        sum(value for _, value in reward_points[-window:]) / window if window else None
    )
    slope = linear_slope(reward_points)

    summary = {
        "tensorboard_run": str(run_dir),
        "scalar_tags": tags,
        "primary_reward_tag": primary_reward,
        "reward_points": len(reward_points),
        "reward_first_window_mean": first_mean,
        "reward_last_window_mean": last_mean,
        "reward_linear_slope_per_step": slope,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    trend = "无法判定"
    if first_mean is not None and last_mean is not None:
        trend = "上升" if last_mean > first_mean else "未上升"
    plot_markdown = "\n\n".join(
        f"### {label}\n\n![{label}]({filename})" for label, filename in plots
    ) or "没有找到可绘制的 scalar tag。"
    report = f"""# LexBrowser WebVoyager 训练报告

## 结论

本报告由完成后的 NeMo RL TensorBoard event 自动生成。Reward 趋势：**{trend}**。
该结论仅按实际日志计算，不对曲线进行人工平滑或挑选 step。

## 运行配置

- 环境：`lexbrowser/webvoyager-no-anti-bot`
- 数据：过滤后的 WebVoyager，600 个真实网页导航任务
- 模型：`/home/wf/models/Qwen3-1.7B`
- 算法：同步 multi-turn GRPO，DTensor v2 LoRA（rank 8，alpha 32）
- 训练：100 optimizer steps；8 prompts/step；8 rollouts/prompt
- 浏览器：Lexmount cloud Chrome/CDP；DOM mode；最多 20 活跃 session
- Reward：无 tool call 为 0；否则由 GLM judge 对完整工具轨迹做二值判定
- TensorBoard run：`{run_dir}`

## Reward 数值摘要

- 主 tag：`{primary_reward or 'N/A'}`
- 记录点数：{len(reward_points)}
- 前窗口均值：{fmt(first_mean)}
- 后窗口均值：{fmt(last_mean)}
- 线性斜率/step：{fmt(slope)}

{plot_markdown}

## 原始产物

- [全部 scalar 数据](training_scalars.csv)
- [机器可读摘要](summary.json)

## 解释边界

真实网站状态会随时间变化。训练跑通和 reward 上升说明 Lexmount Browser + NeMo RL
链路能够产生可学习的真实网站轨迹；要证明相对 Browserbase 的“非劣效”，仍需对同一
checkpoint、同一批任务和同一 judge 做 paired A/B，并报告成功率差值的置信区间。
"""
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
