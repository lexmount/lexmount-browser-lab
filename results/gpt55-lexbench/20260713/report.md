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

## 固定条件

| 项目 | 值 |
| --- | --- |
| Runner | `lexmount/browseruse-agent-bench@bce2c2a17dc2bcf3062b56df4946230c94426cd6` |
| Dataset | `LexBench-Browser / All / 210`，SHA-256 `b2e8626...e2f6b90fe` |
| Agent / model | `browser-use 0.13.4 / gpt-5.5` |
| Judge | `gpt-5.4 / per-task threshold stepwise` |
| 限制 | 40 steps，600 秒，并发 10 |
| Local 环境 | Apple M4 Pro 14 核，48 GB，macOS 15.7.3，Chrome 150.0.7871.102 |

5090 主机不可达，因此 Local arm 使用同一台 Mac 的系统 Chrome。两端在同机顺序执行，
只替换 browser backend。

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
