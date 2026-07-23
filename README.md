# Lexmount Browser Lab

Lexmount Browser与Local Chrome/Playwright的可复现实验仓库。Python实验包名为
`lexbrowserenv-experiments`。Agent、数据集和Judge
尽量使用官方实现；本仓库保存固定实验配置、一键Shell入口、进程归因资源监控和可审计报告。

## 支持范围

| Benchmark | 实验 | Browser backend | 状态 |
|---|---|---|---|
| LexBench-Browser | GPT-5.5 + gpt-5.4 Judge | Lexmount / Local Chrome | 已完成 |
| LexBench-Browser | Qwen3-8B + gpt-5.4 Judge | Lexmount / 5090 Local Chrome | 已完成正式与压力评测 |
| Online-Mind2Web | Qwen3-8B + gpt-5.4 WebJudge | Lexmount / 5090 Local Chrome | 已完成 |
| WebArena-Lite | OpenAI-compatible model | 官方Playwright | runner已实现；Lexmount adapter尚未实现 |

## 安装

```bash
uv sync --extra dev
```

运行凭据通过Shell格式的env文件注入。可从[.env.example](.env.example)复制，并按实验补充
Qwen、Judge、Lexmount、Hugging Face或WebArena站点配置。

## 运行前置条件

Lexmount/NVIDIA训练和线上基准任务的runner必须具备DNS与HTTPS外网访问能力；纯内网集群
不能执行这些任务。运行机器至少需要能够访问NGC、PyPI、Hugging Face、DMX Judge API、
Lexmount API，以及对应的真实WebVoyager/WebArena目标站点。

## 一键运行

### LexBench：Qwen3-8B

同时执行Lexmount和Local的正式质量评测、压力rollout及gpt-5.4 Stage 2：

```bash
./scripts/run_lexbench.sh qwen3-8b \
  --env-file /data/wf/sxh/.env.lexbench \
  --runtime-root /data/wf/sxh \
  --backend all \
  --mode all \
  --stage all
```

完成后会在`<runtime-root>/results/lexbench/<campaign-id>/`生成
`campaign_report.json`与`campaign_report.md`。报告直接读取官方gpt-5.4 summary、
压力Stage 2 aggregate及cgroup/进程树CSV；缺少、覆盖不足或Judge模型不符时会失败，
不会发布部分结果。

真实端到端 smoke（先取官方 All 的固定前10条，分别完成 Qwen3-8B rollout 与 gpt-5.4
Judge；仅验证入口，不可替代 210-task 正式分数）：

```bash
./scripts/run_lexbench.sh qwen3-8b \
  --env-file /data/wf/sxh/.env.lexbench \
  --runtime-root /data/wf/sxh \
  --campaign-id lexbench_smoke_YYYYMMDDTHHMMSSZ \
  --backend all --mode quality --stage all --task-count 10
```

指定campaign或断点续跑：

```bash
./scripts/run_lexbench.sh qwen3-8b \
  --env-file /data/wf/sxh/.env.lexbench \
  --runtime-root /data/wf/sxh \
  --campaign-id lexbench_YYYYMMDDTHHMMSSZ \
  --backend local \
  --mode quality \
  --stage all \
  --resume
```

只检查将要执行的官方命令：

```bash
./scripts/run_lexbench.sh qwen3-8b \
  --env-file /path/to/eval.env \
  --backend all \
  --mode quality \
  --stage all \
  --dry-run
```

### LexBench：GPT-5.5

```bash
./scripts/run_lexbench.sh gpt5.5 \
  --benchmark-repo /path/to/browseruse-agent-bench-worktree \
  --env-file /path/to/eval.env \
  --backend lexmount \
  --phase full \
  --concurrency 10
```

支持`smoke|pilot|full|capacity`；容量模式额外传入`--count N`。

### Online-Mind2Web

```bash
./scripts/run_online_mind2web.sh \
  --env-file /data/wf/sxh/.env.lexbench \
  --runtime-root /data/wf/sxh \
  --backend all \
  --stage all
```

支持`prepare|rollout|judge|report|all`，并可通过`--campaign-id`绑定已有campaign；
campaign ID格式为`YYYYMMDD_HHMMSS`。

真实端到端 smoke 会先验证完整固定300-task官方数据 blob，再固定选其前10条；两后端各执行
Qwen3-8B rollout 与 gpt-5.4 WebJudge，所有产物落在独立 campaign：

```bash
./scripts/run_online_mind2web.sh \
  --env-file /data/wf/sxh/.env.online_mind2web_v2 \
  --runtime-root /data/wf/sxh \
  --campaign-id 20260713_120000 \
  --backend all --stage all --task-count 10
```

### WebArena-Lite

先启动官方WebArena站点环境，并确保`SHOPPING`、`SHOPPING_ADMIN`、`REDDIT`、
`GITLAB`、`MAP`、`WIKIPEDIA`和`HOMEPAGE`均可从runner访问；`--server`不是站点
部署命令。站点不健康时runner会在正式任务前失败。

```bash
./scripts/run_webarena_lite.sh \
  --env-file /path/to/eval.env \
  --backend playwright \
  --runtime-root /data/wf/sxh \
  --server <webarena-server> \
  --start 0 \
  --end 165
```

当前runner只支持官方Playwright backend。站点环境准备脚本为
[`scripts/setup_webarena_lite_env.sh`](scripts/setup_webarena_lite_env.sh)。

## 评估结果摘要

### LexBench-Browser

| 模型 | Backend | Success | Avg steps | Avg e2e |
|---|---|---:|---:|---:|
| GPT-5.5 | Lexmount | **71.43%（150/210）** | 11.84 | 129.83 s |
| GPT-5.5 | Local Chrome | 58.10%（122/210） | 12.10 | **96.73 s** |
| Qwen3-8B | Lexmount | **11.43%（24/210）** | 11.04 | 304.90 s |
| Qwen3-8B | Local Chrome | 8.57%（18/210） | 10.08 | **267.48 s** |

GPT-5.5补测把88条原始Mac Local failure全部放到5090重跑；只恢复10条。即使只把
这10条加回Local，Lexmount仍领先8.57pp，配对bootstrap 95% CI为[+1.43,+15.71]pp。
固定64任务的Lexmount c64相对c10把吞吐从194.3提升到319.9 task/h，但实际active
峰值为39、PSS P95从3.80升到14.30 GiB；Judge差值-3.13pp的95% CI包含0。

Qwen3-8B压力测试中，Lexmount受云端session quota限制，最大可持续并发为20；
Local在本次46 GiB评测cgroup限制下最大可持续并发为60，c80被systemd-oomd终止。
同为c20时，Local吞吐高19.1%，但平均CPU约为Lexmount的9.6倍，进程树PSS
多5.72 GiB，并额外占用5.38 GiB Chrome PSS。

### Online-Mind2Web

| Backend | Success | Avg steps（有效轨迹） | Avg e2e（有效轨迹） |
|---|---:|---:|---:|
| Lexmount | 5.00%（15/300） | 9.29 | **128.34 s** |
| Local Chrome | **5.67%（17/300）** | 9.14 | 158.60 s |

该结果使用官方WebJudge流程，但Judge backbone和最终强制失败策略属于本轮明确偏差，
不是OSU Leaderboard官方验证成绩。

## 报告

- [全部评估报告索引](docs/eval_reports/README.md)
- [Qwen3-8B LexBench汇总](docs/eval_reports/lexbench/README.md)
- [Qwen3-8B LexBench压力与资源报告](docs/eval_reports/lexbench/stress_results.md)
- [Online-Mind2Web汇总](docs/eval_reports/online-mind2web/README.md)
- [GPT-5.5 LexBench报告](results/gpt55-lexbench/20260713/report.md)
- [GPT-5.5 LexBench整夜稳定性综合](results/gpt55-lexbench/overnight-20260713/stage4/report.html)

## H100 GRPO training reproduction

`training/h100/` is a self-contained H100/CUDA reproduction package for the
validated Browser-RL recipe (Qwen3-8B + verl GRPO + NeMo-Gym browser sidecar +
Lexmount browser + WebVoyager tasks, 60 steps). It vendors its own pinned
runtime and task data and touches nothing else in this repository. Start at
[training/h100/README.md](training/h100/README.md); porting provenance is in
[training/h100/PORTING.md](training/h100/PORTING.md).

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/lexbrowser_eval/` | Python评测与资源监控module |
| `scripts/` | 仅Shell入口 |
| `experiments/` | 固定配置和任务集合 |
| `docs/eval_reports/` | Qwen3-8B正式评估报告 |
| `results/gpt55-lexbench/` | main已有GPT-5.5可审计结果 |
| `training/h100/` | 自包含的H100/CUDA GRPO复现包（含vendored runtime与数据） |
| `tests/` | 与src package对应的测试 |

原始轨迹、截图和运行日志保留在运行机器，不提交Git。Linux资源报告使用cgroup
CPU/PSS/Chrome PSS；macOS GPT-5.5 fallback使用进程树CPU/RSS，两种内存口径不混算。
