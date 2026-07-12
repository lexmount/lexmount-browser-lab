# LexBrowserEnv

Lexmount Browser 与本地 Chrome 的可复现实验仓库。浏览器 Agent、数据集和 Judge
仍由 [`browseruse-agent-bench`](https://github.com/lexmount/browseruse-agent-bench)
提供；本仓库只保存实验协议、固定配置、采样工具和可审计结果。

## 当前实验

`gpt-5.5 / browser-use / LexBench-Browser All`，在相同任务、模型、Agent 参数和
Judge 下，仅替换浏览器后端：

- `lexmount`：远端隔离浏览器
- `local`：5090 主机上的本地 Chrome

实验回答两个问题：

1. Lexmount 与本地 Chrome 的任务质量和运行稳定性是否一致。
2. Lexmount 能否把 Chrome 的 CPU、内存和进程压力移出 rollout 主机。

完整口径见 [`experiments/gpt55-lexbench/README.md`](experiments/gpt55-lexbench/README.md)。
当前 preflight 证据见
[`results/gpt55-lexbench/preflight-20260712.md`](results/gpt55-lexbench/preflight-20260712.md)。

## 仓库边界

| 内容 | 所在位置 |
| --- | --- |
| Agent、benchmark、Judge | `browseruse-agent-bench` |
| 实验配置和任务集合 | `experiments/` |
| 并发探针和资源采样 | `scripts/` |
| 汇总指标和最终报告 | `results/` |
| 原始轨迹、截图、日志 | 运行机器，不提交 Git |

## 快速检查

```bash
uv sync --extra dev

# 不调用模型，只验证 Lexmount 同时创建 session 的实际上限。
uv run python scripts/probe_lexmount_sessions.py \
  --env-file /path/to/browseruse-agent-bench/.env \
  --profile en \
  --count 40

# 生成固定、分层的容量任务集合。
uv run python scripts/select_tasks.py \
  --dataset /path/to/browseruse-agent-bench/browseruse_bench/data/LexBench-Browser/task.jsonl \
  --count 80
```

Linux 5090 runner 或 macOS fallback 上的配对运行入口：

```bash
./scripts/run_benchmark.sh \
  --benchmark-repo /path/to/browseruse-agent-bench-worktree \
  --env-file /path/to/.env \
  --backend lexmount \
  --phase smoke \
  --concurrency 2
```

凭证只从外部 `.env` 注入。仓库内的配置只包含环境变量引用。

Linux 使用 cgroup CPU/PSS/Chrome PSS；macOS 使用同进程树 CPU/RSS/Chrome RSS。
两种口径不会混在同一张资源对比表中。
