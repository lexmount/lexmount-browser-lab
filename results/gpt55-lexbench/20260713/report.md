# GPT-5.5 × LexBench-Browser 全量配对结果

## 结论

在 210 条相同任务、GPT-5.5、browser-use 0.13.4、并发 10 和同一 GPT-5.4
Judge 下，Lexmount 成功 150 条（71.43%），本地 Chrome 成功 122 条（58.10%）。
配对差值为 **+13.33 个百分点**，95% bootstrap CI 为
**[+6.19, +20.48] 个百分点**，通过预先定义的 5 个百分点非劣判定，并在本轮中
表现为统计上清晰的优势。

这轮结果支持三个判断：

1. Lexmount 的 browser session 层在 c10 下稳定，210/210 条均形成轨迹，session
   创建失败为 0，结束后 active session 为 0。
2. 端到端质量优势主要来自 browser backend 的站点访问环境差异。它包含出口 IP、
   地区、指纹和页面可达性，不能解释成远端 Chromium 内核本身比本地 Chrome 更强。
3. Lexmount 把 Chrome 进程压力移出了 runner，但更慢：runner RSS P95 从
   16.95 GiB 降到 4.61 GiB，吞吐从 344.3 降到 241.2 task/h。

后续复核没有把一次 Local 失败当成稳定标签：88 条原始 Mac Local failure 全部在
5090 上重跑。只恢复 10 条；即使只给 Local 加回这 10 条、且保留原始全部成功，
Lexmount 仍领先 +8.57pp，配对 bootstrap 95% CI 为 [+1.43, +15.71]pp。这个结果是
偏向 Local 的敏感性分析，不是第二轮独立成功率估计。

同一 64 任务的 Lexmount c64 补测将吞吐从 194.3 提到 319.9 task/h，但实际 active
峰值为 39 而不是 64，PSS P95 从 3.80 升到 14.30 GiB。Judge 从 43/64 变为
41/64，配对差值 -3.13pp 的 95% CI 为 [-14.06, +7.81]pp，没有清晰质量退化证据。

## 固定条件

| 项目 | 值 |
| --- | --- |
| Runner | `lexmount/browseruse-agent-bench@bce2c2a17dc2bcf3062b56df4946230c94426cd6` |
| Dataset | `LexBench-Browser / All / 210`，SHA-256 `b2e8626...e2f6b90fe` |
| Agent / model | `browser-use 0.13.4 / gpt-5.5` |
| Judge | `gpt-5.4 / per-task threshold stepwise` |
| 限制 | 40 steps，600 秒，并发 10 |
| Local 环境 | Apple M4 Pro 14 核，48 GB，macOS 15.7.3，Chrome 150.0.7871.102 |

原始 210 条全量运行时 5090 旧地址不可达，因此 Local arm 使用同一台 Mac 的系统
Chrome。后续拿到新地址后，另在 5090 上完成 failure-only 与容量复核；原始配对数字
不被覆盖。

## 5090 failure-only 复核

原始 Mac Local 的 88 条失败在 Ubuntu 22.04、Chrome 140、i9-14900K、125.6 GiB
内存上以 c10 全部重跑并重新 Judge：

| 指标 | 5090 Local rerun |
| --- | ---: |
| Planned / trajectory / judged / success | 88 / 88 / 88 / 10 |
| Agent steps mean / P50 / P95 | 13.81 / 9.5 / 35 |
| E2E 秒 mean / P50 / P95 | 133.85 / 93.36 / 408.95 |
| Timeout / network / unhandled 信号 | 6 / 16 / 1 |
| 平均 CPU cores | 1.00 |
| PSS mean / P95 / max | 3.96 / 5.56 / 6.25 GiB |
| Chrome PSS mean / P95 / max | 1.73 / 2.74 / 3.39 GiB |
| cgroup memory.current mean / P95 / max | 5.25 / 7.20 / 8.01 GiB |
| cgroup memory.peak | 8.05 GiB |
| Host available 最小值 | 71.62 GiB |
| 吞吐 | 204.1 task/h |
| Agent token / cost | 11.80M / $52.13 |
| Judge token | 0.714M |
| GPU util / idle / memory / power（host-wide） | 4.29% / 95.71% / 22,255 MiB / 30.49 W |

恢复的 10 条为 18、22、24、82、124、288、298、302、3014、3020；其中 6 条原来
Lexmount-only，4 条原来双方都失败。51 条原始英文 failure 本轮仍为 0/51。原始
31 条 E1/E2/E3 环境类本地 loser 中只有 2 条恢复，29 条仍失败。

12 条 smoke 先恢复 task 114、3017，但它们在 88 条完整复测中再次失败；task 22
反而从 smoke 失败翻为成功，12 条交集没有一条连续两次成功。这证明轮次波动存在，
但它不足以解释原始 28 条净差。完整分解见 [`log-audit.md`](log-audit.md)。

复测当下，Mac 与 5090 实测为同一公网 IPv4、北京联通 AS4808 和同一地区；公网 IP
不提交。原始全量运行没有保存出口快照，因此不能倒推昨夜也一定相同。

## Lexmount c10 / c64 容量复核

同一台 5090 runner、同一组 `capacity64.txt` 的 64 条分层任务，顺序执行 Lexmount
c10 与 c64。两点均为 64/64 形成轨迹、session 创建失败 0、无 OOM/资源护栏，结束后
en/zh project 的 active session 均为 0。

| 指标 | c10 | c64 | 变化 |
| --- | ---: | ---: | ---: |
| 请求并发 | 10 | 64 | 6.4x |
| 实际 active session | live snapshot 10（en 4 / zh 6） | mean 16.04 / P95 36 / max 39 | 峰值未到 64 |
| Judge 成功 | 43/64（67.19%） | 41/64（64.06%） | -3.13pp |
| 同题单方成功 | c10-only 8 | c64-only 6 | 净 -2 |
| 同题双方成功 / 双方失败 | 35 / 15 | 35 / 15 | 配对 CI 含 0 |
| E2E 秒 mean / P50 / P95 | 130.58 / 92.52 / 417.49 | 173.40 / 132.56 / 482.71 | 单任务延迟上升 |
| Timeout / network / unhandled 信号 | 0 / 3 / 3 | 1 / 3 / 4 | 无 session create failure |
| Rollout 时长 | 19.77 min | 12.00 min | -39.27% |
| 吞吐 | 194.3 task/h | 319.9 task/h | +64.7% |
| 平均 CPU cores | 0.48 | 0.70 | +47.1% |
| PSS mean / P95 / max | 2.55 / 3.80 / 4.00 GiB | 6.24 / 14.30 / 14.71 GiB | P95 +276% |
| memory.current mean / P95 / max | 2.95 / 4.11 / 4.29 GiB | 6.75 / 14.47 / 14.91 GiB | max +248% |
| cgroup memory.peak | 4.31 GiB | 14.94 GiB | +247% |
| Host available 最小值 | 79.46 GiB | 68.43 GiB | -11.03 GiB |
| Agent token / cost | 9.67M / $41.89 | 8.68M / $40.75 | 单轮轨迹差异 |
| Judge token | 0.556M | 0.549M | 基本一致 |
| GPU util / idle（host-wide） | 2.83% / 97.17% | 0.01% / 99.99% | 不作因果解释 |
| GPU memory / power（host-wide） | 22,859 MiB / 29.83 W | 21,892 MiB / 13.17 W | 被其他负载污染 |
| vLLM running / waiting | N/A | N/A | GPT-5.5 为远端 API |

c64 的 session sampler 从 rollout 启动开始采样 144 次、无采样错误；c10 sampler
入口错误导致完整曲线缺失，因此这里只保留运行中的 API 直查快照，不把它写成均值。
c64 的 en 峰值 23、zh 峰值 34，二者发生在不同时间；中文 session 先就绪，英文
session 后续补齐时已有中文任务结束，所以总 active 峰值只有 39。

单 project 的 64 路原始探针也印证启动瓶颈：180 秒内 en 创建 25/64，zh 创建
48/64。旧探针退出后仍有延迟 session 激活，其即时 `sessions_after=0` 字段无效；已
手工归零，并修复为“异常 ID 直接删除 + 延迟轮询 + 最终基线校验”。因此当前最准确的
容量表述是：**配置允许请求 c64，端到端吞吐明显提升，但本轮没有实现 64 个同时 active
session；可观测峰值为 39。**

5090 上有其他 GPU workload，且 GPT-5.5 由远端 API 提供；上表 GPU 数据是整机采样，
只为指标完整性保留，不能归因于 Local Chrome、Lexmount runner 或模型推理。

c64 与 c10 的 Judge 配对差值为 -3.13pp，100,000 次 bootstrap 95% CI
`[-14.06, +7.81]pp`；单轮数据不支持“c64 明显降低质量”，也不足以证明等价。c64
提高批量吞吐的同时，单任务 E2E mean 从 130.58 秒升到 173.40 秒，适合离线批处理，
不应直接用于追求单任务低延迟的交互流量。

## 配对质量

| 指标 | Lexmount | Local Chrome |
| --- | ---: | ---: |
| Judge 成功 | 150/210（71.43%） | 122/210（58.10%） |
| 同题双方成功 | 104 | 104 |
| 同题单方成功 | 46 | 18 |
| 同题双方失败 | 42 | 42 |
| Agent steps mean / P50 / P95 | 11.84 / 9 / 33 | 12.10 / 10 / 31 |
| E2E 秒 mean / P50 / P95 | 129.83 / 93.72 / 381.50 | 96.73 / 62.10 / 271.73 |

分层结果：

| 分层 | Lexmount | Local Chrome | 差值 |
| --- | ---: | ---: | ---: |
| en / T1 | 55/84（65.48%） | 34/84（40.48%） | +25.00pp |
| en / T2 | 7/8（87.50%） | 7/8（87.50%） | 0.00pp |
| zh / T1 | 81/110（73.64%） | 73/110（66.36%） | +7.27pp |
| zh / T2 | 7/8（87.50%） | 8/8（100.00%） | -12.50pp |

T2 每个语言只有 8 条，不单独外推。

## 因果审计

64 条单边成功任务的 loser 主分类如下：

| Loser 主分类 | Lexmount-only | Local-only | 净差 |
| --- | ---: | ---: | ---: |
| E1/E2/E3 站点或访问环境 | 31 | 5 | +26 |
| M1-M4 Agent 规划/证据 | 15 | 10 | +5 |
| H1 Harness | 0 | 3 | -3 |
| 合计 | 46 | 18 | +28 |

因此总净优势 28 条中，有 26 条与站点/访问环境主分类的不对称同方向；去掉 E 类后只剩
净 +2 条。31 个本地 E 类 loser 中，25 个在原始 action/result log 中直接出现
captcha、HTTP access denial 或 `net::ERR_*` 信号。这个分解是事后诊断，不是随机化后
的“调整成功率”。

另取 12 条机制样本做两轮反向顺序复测，加原始运行共 3 次/arm。ASOS task 97 与 3DM
task 3008 为 Lexmount `3/3`、Local `0/3`；58.com task 180 则为 Lexmount `0/3`、
Local `3/3`。Scholar、Crunchyroll、GameSpot 等存在轮次翻转，证明单次站点状态、Agent
路径和 Harness 仍有明显随机性。该样本是按机制选择的，不能作为新的总体成功率估计。

完整证据和截图哈希见 [`log-audit.md`](log-audit.md)，逐题数据见
[`paired-log-audit.json`](paired-log-audit.json) 与
[`mechanism12-replays.json`](mechanism12-replays.json)。

## 稳定性

| 指标 | Lexmount | Local Chrome |
| --- | ---: | ---: |
| Planned / trajectory / judged | 210 / 210 / 210 | 210 / 210 / 210 |
| Agent terminal state done / error / timeout | 199 / 10 / 1 | 206 / 4 / 0 |
| Runner task process zero / nonzero exit | 200 / 10 | 206 / 4 |
| Session 创建失败 | 0 | 0 |
| Timeout 信号 | 8 | 11 |
| Navigation / network 信号 | 6 | 18 |
| Unhandled error 信号 | 11 | 4 |
| 资源护栏触发 | 0 | 0 |

错误信号可能在同一任务重叠。Agent terminal state 与 runner process exit 是不同口径；
Lexmount 的 199 done、10 error、1 timeout 完整覆盖 210 条。Lexmount 没有 session 层
容量失败，但不能表述为无任务级失败；10 个 task 子进程 nonzero exit 仍保留在正式
结果中，未按结果好坏重跑。

Judge 失败归因中，站点环境类 `E1`（bot defense）、`E2`（access barrier）、`E3`
（site limitation）合计为 Lexmount 19/60、本地 55/88。这是 Judge 的归因信号，
不是对单个站点根因的确定性证明，但与本地 navigation/network 信号更高相互印证。

## 资源与成本

| 指标 | Lexmount runner | Local runner + Chrome |
| --- | ---: | ---: |
| 平均 CPU cores | 0.37 | 1.72 |
| RSS mean / P95 / max | 3.74 / 4.61 / 5.14 GiB | 13.87 / 16.95 / 23.20 GiB |
| Chrome RSS mean / P95 / max | 0 / 0 / 0 GiB | 10.79 / 13.72 / 19.73 GiB |
| Host available 最小值 | 12.69 GiB | 9.06 GiB |
| Rollout 时长 | 52.25 min | 36.59 min |
| 吞吐 | 241.2 task/h | 344.3 task/h |
| Agent token / cost | 26.55M / $119.94 | 24.31M / $104.55 |
| Judge token / cost | 1.76M / $7.34 | 1.74M / $7.35 |

macOS 采集的是进程树 RSS，不是 Linux cgroup PSS。Lexmount 列只覆盖本机 runner
控制进程，不包含远端浏览器宿主机资源，因此可以证明本机压力被移出，不能据此声称
Lexmount 服务端总资源更低。GPT-5.5 使用远端 API，本轮 GPU、vLLM 指标为 N/A。

## 边界

- 210 条总体结果仍是单次顺序运行；12 条机制复测只验证选中案例，不能替代全量多轮复测。
- Local 使用 M4 Pro fallback，不是原计划的 5090 Linux；没有 cgroup PSS、GPU 或功耗数据。
- 两端出口、地区、浏览器指纹和运行环境不同，因此当前实验评估的是完整 backend 服务，
  不是浏览器内核的隔离 A/B。
- Agent 没有固定 seed；Judge temperature 为 1.0 且每条只判一次。64 个单边成功任务中
  有 12 个 loser 距阈值不超过 10 分，阈值附近标签需要重复 Judge 才能估计方差。
- Judge 当前串行执行，不代表端到端 Judge 容量；机制复测有 1 条因 GPT-5.4 429 未获得
  真实 Judge 分数，已按 missing 排除而不是计 0 分。

机读数据见 [`metrics.json`](metrics.json)，实验身份见 [`manifest.json`](manifest.json)。
