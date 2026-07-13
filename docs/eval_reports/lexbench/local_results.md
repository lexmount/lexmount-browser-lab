# LexBench 评测报告：Qwen3-8B + Local Chrome

## 结论摘要

本次使用官方 LexBench-Browser `All`（210 tasks）、5090 本地 Chrome/Xvfb、rollout concurrency=10。正式 gpt-5.4 stepwise Judge 成功率为 **8.57%（18/210）**。主要失败来自模型规划、页面理解、错误恢复及模型服务/结构化输出；Local Chrome 还出现了较多导航和本机网络异常，因此不能把全部失败归因于 Qwen。

## 正式指标

| 指标 | 结果 | 口径 |
|---|---:|---|
| Success | **8.57%**（18/210） | gpt-5.4 stepwise 分数达到每任务官方阈值 |
| Avg steps | **10.08** | 206 条具有 Agent 指标；4 条为外部终止产生的 interrupted 记录 |
| Avg e2e | **267.48 s** | 官方 `end_to_end_ms` 汇总换算为秒，206 条 |
| 平均 Judge 分数 | 21.72 / 100 | 210 条 |

正式产物：

```text
/data/wf/sxh/.lexbench/browseruse-agent-bench/experiments/LexBench-Browser/All/browser-use/qwen3_8B/20260710_193432/tasks_eval_result/task_gpt-5.4_per_task_threshold_stepwise_eval_results.json
/data/wf/sxh/.lexbench/browseruse-agent-bench/experiments/LexBench-Browser/All/browser-use/qwen3_8B/20260710_193432/tasks_eval_result/task_gpt-5.4_per_task_threshold_stepwise_summary.json
```

## c10 rollout 资源效率

rollout 被外部 `SIGINT` 中断后以同一 timestamp 和 `--skip-completed` 恢复。总有效运行时间约 **5,786 s（96.43 min）**，据此计算全程吞吐；但恢复段没有连续 sidecar，因此 CPU/GPU/内存只能报告中断前 2,306 秒的连续观测窗口，覆盖约 **39.85%** 的有效运行时间。

| 指标 | 结果 | 覆盖说明 |
|---|---:|---|
| 并发 | 10 tasks | 全程一致 |
| 吞吐 | **0.0363 task/s = 2.18 task/min = 130.66 task/h** | 两段有效运行时间，排除停机空窗 |
| 5090 Host CPU | **平均 22.83%**；峰值 68.90% | 连续窗口，baseline 5.86% |
| 5090 内存 | **平均 50.50 GiB**；峰值 53.16 GiB | 连续窗口，baseline 约 44.99 GiB |
| Local Chrome RSS | **平均 16.30 GiB**；峰值 19.50 GiB | 连续窗口 |
| GPU SM（两卡平均） | **97.78%** | 连续窗口；主要为 Qwen3-8B vLLM 推理 |
| GPU idle（两卡平均） | baseline 100% → rollout 1.53%，降低 **98.47 个百分点** | 连续窗口 |

CPU、内存和 GPU 数字不是完整 210-task 全程均值，不能用于精确容量结论；最终 local/云浏览器资源对比必须保留这一缺口。

## 完成状态与 Judge 结果

| Agent 终态 | Judge 通过 | Judge 未通过 | 合计 |
|---|---:|---:|---:|
| 正常结束（`done`） | 18 | 119 | 137 |
| 600 秒 timeout | 0 | 62 | 62 |
| runtime error | 0 | 7 | 7 |
| 外部终止 interrupted | 0 | 4 | 4 |
| 合计 | 18 | 192 | 210 |

- 206 条产生非空最终回答；4 条 interrupted 没有最终回答。
- Agent 自报成功 45 条，其中 14 条 Judge 通过、31 条未通过，存在明显的完成质量高估。
- 137 条正常结束中仍有 119 条 Judge 未通过，说明核心问题仍是任务要求覆盖不全、证据不足或结论错误，而不只是“没有输出”。

## 官方失败分类

| 分类 | 数量 | 占 192 个失败 | 含义 |
|---|---:|---:|---|
| M1 | 77 | 40.10% | 任务规划错误或关键步骤遗漏 |
| M2 | 32 | 16.67% | 页面理解、元素定位或 grounding 错误 |
| M4 | 27 | 14.06% | 错误恢复不足、循环或停滞 |
| M6 | 15 | 7.81% | 模型服务、上下文或结构化输出异常 |
| M3 | 3 | 1.56% | 最终答案与页面证据不一致或证据不足 |
| E3 | 21 | 10.94% | 站点能力或页面本身限制 |
| E1 | 6 | 3.12% | 反爬、风控或机器人防护 |
| E2 | 3 | 1.56% | 登录、权限、地区等访问障碍 |
| H1 | 4 | 2.08% | 评测执行或浏览器工具链缺陷 |
| not_evaluated | 4 | 2.08% | 无法形成可评估证据 |

模型类 M1–M6 共 **154/192（80.21%）**；环境/站点类 E1–E3 共 **30/192（15.63%）**；H1 与 not_evaluated 共 8 条。因此低成功率主要与模型能力相关，但 Local Chrome 网络和浏览器异常是不可忽略的干扰因素。

## 错误与轨迹诊断

- **62 条**达到官方 600 秒任务超时，是最大的“无正常结果”来源。
- **7 条** runtime error：3 条明确为导航失败，其余为本地执行错误。
- 外部终止时生成 **4 条 interrupted**，均保留且按用户要求未重跑；这四条 `browser_id` 为空，也是唯一没有最终回答的记录。
- Local Chrome 日志中，100 个任务出现过 `Navigation failed`，24 个任务出现过 `ERR_NETWORK_CHANGED`；未发现 Browser closed 或 Target closed。说明浏览器整体可运行，但本机网络/导航路径明显不稳定。
- Agent API logs 共记录 8 次 LLM step failure：6 次结构化输出/JSON schema 失败、2 次 40,960 context 超限；多数任务随后恢复，部分最终 timeout。
- task 2004 的截图触发内容策略，官方无图降级后成功评分为 8 分。task 43、93、124、127 被记录为 `not_evaluated`。

## Stage 2 审计

- Judge：gpt-5.4、stepwise、官方每任务 `score_threshold`。
- 为满足 Judge concurrency=5，将 210 条轨迹按 task_id 确定性分成 5 个隔离 shard，每 shard 42 条；每个 shard 调用同一官方 evaluator，官方源码未修改。
- 隔离 dry-run 为 gpt-5.4、1/1；五个 shard 完成后校验 210 个唯一 task_id，再原子生成正式 JSONL。
- 失败分类单独以 concurrency=5 运行；最终 192 个失败全部分类完成。
- 原 210 个 `result.json` 的 hash 与 mtime 校验为 210/210 未改变。
- 官方 checkout commit：`ccd5fcbdfb975257b2ce38161dc9bc2ab294b420`，tracked worktree clean。

## 与 Lexmount Browser 的初步对比

| 指标 | Lexmount | Local Chrome | 差异 |
|---|---:|---:|---:|
| Success | 11.43% | 8.57% | Local -2.86 pp |
| Avg steps | 11.04 | 10.08 | Local -0.96 |
| Avg e2e | 304.90 s | 267.48 s | Local 快约 12.3% |
| 吞吐 | 109.23 task/h | 130.66 task/h | Local +19.6% |
| Host CPU | 18.82% | 22.83%* | Local +4.01 pp |
| Host 内存 | 47.82 GiB | 50.50 GiB* | Local +2.68 GiB |
| 本地浏览器 RSS | 0 | 16.30 GiB* | Local 额外占用 |

`*` Local 资源数字仅覆盖 39.85% 的连续窗口，因此资源差异只能作为方向性证据。质量对比还存在两个审计边界：Lexmount 最早保留的 38 条使用旧 32K context，而 Local 全程使用 40,960；Lexmount Judge 串行、Local Judge 使用五分片并发 5，但两者的官方 prompt、评分器和阈值相同。206 个可比较任务的 system prompt hash 为 206/206 一致。

综合判断：Local Chrome 的质量分数低于 Lexmount，虽然完成速度更快，但消耗了约 16 GiB 本地浏览器内存，并暴露出明显的导航与 `ERR_NETWORK_CHANGED` 风险。当前证据支持“Lexmount 减少 5090 浏览器资源占用且质量不劣于 Local”，但由于 Local 恢复段资源 sidecar 缺失，不能据此宣称完整容量优势已经得到严格量化。
