# LexBench：Lexmount Browser 与 Local Chrome 对比

正式质量评测对象均为 Qwen3-8B、LexBench-Browser `All`（210 tasks）、rollout concurrency=10，并使用 gpt-5.4 stepwise Judge。详细报告见 [Lexmount Browser](lexbrowser_results.md) 和 [Local Chrome](local_results.md)；进程归因和容量结论见 [高并发压力测试](stress_results.md)。

## 1. 官方质量指标

| 指标 | Lexmount Browser | Local Chrome | Local 相对 Lexmount |
|---|---:|---:|---:|
| Success | **11.43%（24/210）** | **8.57%（18/210）** | **-2.86 pp** |
| Avg steps | 11.04 | 10.08 | -0.96 |
| Avg e2e | 304.90 s | 267.48 s | -37.42 s（快 12.3%） |

## 2. 同并发 c20 进程归因资源效率

以下数据来自固定 20 任务的压力测试，CPU 和 PSS 仅统计评测 cell 的 cgroup/进程树，不包含 5090 上其他进程；因此不与上面的 All 210 质量指标混算。

| 指标 | Lexmount Browser | Local Chrome | Local 相对 Lexmount |
|---|---:|---:|---:|
| 吞吐 | 100.06 task/h | **119.21 task/h** | +19.1% |
| 平均 CPU | **0.25 cores** | 2.39 cores | 9.6× |
| 评测进程树 PSS | **3.92 GiB** | 9.64 GiB | +5.72 GiB |
| Local Chrome PSS | **0 GiB** | 5.38 GiB | +5.38 GiB |
| cgroup 内存峰值 | **5.24 GiB** | 12.29 GiB | +7.05 GiB |
| 最大可持续并发 | 20（session quota） | **60**（c80 被 systemd-oomd 终止） | +40 tasks |

GPU 指标包含批准的外部 Qwen concurrency=1，仅作观察，不用于容量结论。

## 3. 失败任务对比

| 失败维度 | Lexmount Browser | Local Chrome | 简要判断 |
|---|---:|---:|---|
| Judge 未通过 | 186（88.57%） | 192（91.43%） | Local 多 6 条失败 |
| 正常 `done` 但 Judge 未通过 | 133 | 119 | 两端主要问题都是“有结果但质量不足” |
| 未正常结束 | 54 | 73 | Local 的运行稳定性更差 |
| 600 秒 timeout | 47 | 62 | 两端都有循环/停滞，Local 更明显 |
| runtime error / interrupted | 7 / 0 | 7 / 4 | Local 额外包含外部终止的 4 条 interrupted |
| 模型类失败 | 142/186（76.34%） | 154/192（80.21%） | 两端均以规划、页面理解和恢复能力为主 |
| 环境/站点类失败 | 40/186（21.51%） | 30/192（15.63%） | Lexmount 更多被官方归为站点/访问障碍 |
| Harness / not evaluated | 4 | 8 | Local 的执行与可评估证据问题更多 |
| 主要异常信号 | 40/47 timeout 轨迹存在重复动作；无全局云浏览器故障 | 100 个任务出现导航失败，24 个出现 `ERR_NETWORK_CHANGED` | Local Chrome 本机网络/导航不稳定是额外干扰因素 |

总体上，Lexmount 的 All 210 成功率更高；同为 c20 时显著降低 5090 侧 CPU 和内存，但当前云端 session quota 将最大并发限制在 20。Local Chrome 吞吐和容量更高，但资源消耗、超时与本机网络异常更多，且 c80 已触发 systemd-oomd。
