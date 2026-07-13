# LexBench 评测报告：Qwen3-8B + Lexmount Browser

## 结论摘要

本次使用官方 LexBench-Browser `All`（210 tasks）和官方 stepwise Judge 流程。Qwen3-8B 在 Lexmount 云浏览器后端的正式成功率为 **11.43%（24/210）**。失败主要来自模型任务规划、页面理解和错误恢复；站点限制、反爬和访问障碍也占一定比例。正式 rollout 期间未发生全局 Qwen、Lexmount 或资源服务中断。

## 正式指标

| 指标 | 结果 | 口径 |
|---|---:|---|
| Success | **11.43%**（24/210） | gpt-5.4 stepwise 分数达到每任务官方阈值 |
| Avg steps | **11.04** | 209 条具有 Agent 指标；task 2024 未成功 Judge |
| Avg e2e | **304.90 s** | 官方 `end_to_end_ms` 汇总换算为秒，209 条 |
| 平均 Judge 分数 | 25.71 / 100 | 210 条，含 task 2024 的合成 0 分 |

正式 summary：

```text
/data/wf/sxh/.lexbench/browseruse-agent-bench/experiments/LexBench-Browser/All/browser-use/qwen3_8B/20260710_162153/tasks_eval_result/task_gpt-5.4_per_task_threshold_stepwise_summary.json
```

## c10 rollout 资源效率

只统计正式 Lexmount rollout phase；合并中断前与 `--skip-completed` 续跑两段，排除 baseline、中断空窗、临时 sample10、错误 glm-5.2 Judge 和正式 gpt-5.4 Judge。

| 指标 | 结果 |
|---|---:|
| 并发 | 10 tasks |
| 有效 rollout 时间 | 6,921 s（115.35 min） |
| 吞吐 | **0.0303 task/s = 1.82 task/min = 109.23 task/h** |
| 5090 Host CPU | **平均 18.82%**；P95 25.89%；峰值 46.16% |
| 5090 内存 | **平均 47.82 GiB**；峰值 49.56 GiB |
| Runner RSS | 平均 3.53 GiB；峰值 4.36 GiB |
| 本地浏览器 RSS | **0 GiB**（浏览器运行在 Lexmount 云端） |
| GPU SM（两卡平均） | **96.43%** |
| GPU idle（两卡平均） | baseline 98.33% → rollout 2.76%，降低 **95.57 个百分点** |

GPU 高占用来自 5090 上的 Qwen3-8B vLLM 推理，不是云浏览器。当前只能报告 Lexmount 的绝对指标；与 local Chrome 的资源效率差异需等待 local All/c10 使用同一口径完成后再下结论。

## 完成状态与 Judge 结果

| Agent 终态 | Judge 通过 | Judge 未通过 | 合计 |
|---|---:|---:|---:|
| 正常结束（`done`） | 23 | 133 | 156 |
| 未正常结束（timeout/error） | 1 | 53 | 54 |
| 合计 | 24 | 186 | 210 |

- 210/210 都生成了 `result.json`，且 `answer` 字段非空；但 54 条只是超时或错误后的部分/兜底结果，不能视为正常完成。
- Agent 自报成功 50 条，其中只有 17 条被官方 Judge 判定通过，33 条属于“自报成功但证据或完成质量不足”。
- 156 条正常结束中仍有 133 条 Judge 未通过，说明主要问题不是“完全没输出”，而是输出未满足任务要求、证据不充分或关键步骤遗漏。

## 官方失败分类

| 分类 | 数量 | 占 186 个失败 | 含义 |
|---|---:|---:|---|
| M1 | 88 | 47.31% | 任务规划错误或关键步骤遗漏 |
| M2 | 32 | 17.20% | 页面理解、元素定位或 grounding 错误 |
| M4 | 16 | 8.60% | 遇到错误后恢复不足、循环或停滞 |
| M3 | 6 | 3.23% | 最终答案与页面证据不一致或证据不足 |
| E3 | 23 | 12.37% | 站点能力或页面本身限制 |
| E1 | 12 | 6.45% | 反爬、风控或机器人防护 |
| E2 | 5 | 2.69% | 登录、权限、地区等访问障碍 |
| H1 | 3 | 1.61% | 评测执行或浏览器工具链缺陷 |
| not_evaluated | 1 | 0.54% | Judge 内容策略拒绝后仍无法评分 |

模型类 M1–M4 共 **142/186（76.34%）**；环境/站点类 E1–E3 共 **40/186（21.51%）**。因此低成功率主要由模型规划、页面理解和恢复能力造成，但不能把所有失败都归因于 Qwen。

## 未正常结束轨迹诊断

54 条未正常结束任务中：

- **47 条**达到官方 600 秒任务超时；其中 40/47 出现同类动作重复至少 3 次，25/47 重复导航，14/47 多次使用失效元素，10/47 连续等待，7/47 重复提取但没有有效推进。模式以循环、停滞和恢复失败为主。
- **5 条**因点击动作 watchdog 15 秒超时结束。
- **2 条**因导航失败结束。
- 最终结果中没有上下文长度、JSON schema、限流或 OOM 作为终止原因。早期旧 32K 上下文配置曾在 task 96/122/180 出现 11 次上下文错误，但三条任务都继续运行并产出结果；续跑切换到 40,960 context 后未再出现该错误。

## Judge 与评测异常

- 正式 Judge 为 **gpt-5.4**；误启动的 glm-5.2 仅产生 5 条记录，已停止并保留为无效审计产物，未计入正式结果。
- task 2007 的网络安全提示触发官方安全降级后成功评分；task 48 的截图内容策略触发官方无图降级后成功评分。
- task 2024 在安全降级重试后仍被内容策略拒绝，官方补录为 `not_evaluated` 失败。这是唯一没有获得实际 Judge 分数的任务。
- 断点续跑前已有的 38 个 `result.json` 经 hash 与 mtime 校验均未改写；正式 preservation check 为 38/38 通过。

## 配置与审计口径

- 官方 checkout commit：`ccd5fcbdfb975257b2ce38161dc9bc2ab294b420`，tracked worktree clean。
- 数据集：LexBench-Browser `All`，210 tasks；Agent：Qwen3-8B；Browser：Lexmount；rollout concurrency=10。
- 官方 browser-use 配置：`max_steps=40`、task timeout=600 s、`flash_mode=true`、`use_vision=false`；Agent 单次输出上限由当前依赖默认为 4,096 tokens，vLLM context 为 40,960。
- Judge：gpt-5.4、stepwise、每任务官方 `score_threshold`。资源采样间隔 1 秒，sidecar 最大间隔 2 秒，无监控断流。

综合判断：当前 Qwen3-8B 的主要短板是长程任务规划、页面状态理解和失败恢复；Lexmount 后端在本轮没有出现全局连接或容量故障，并显著避免了 5090 上的本地浏览器内存占用。最终的“云浏览器相对 local Chrome”结论必须等待 local All/c10 和两组压力测试完成后统一比较。
