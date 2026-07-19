# GPT-5.5 × LexBench Browser Stability

实验来源：

- [0709 Lexmount Browser 接入 Nemo-gym 实验计划](https://lexmount.notion.site/0709-lexmount-browser-nemo-gym)
- [上一轮 Lexmount Browser 与 Local Chrome 评测结果](https://app.notion.com/p/399e7e998fb28085b947c3efaa7859f3)

## 固定条件

| 项目 | 值 |
| --- | --- |
| Benchmark runner | `lexmount/browseruse-agent-bench@bce2c2a17dc2bcf3062b56df4946230c94426cd6` |
| Dataset | `LexBench-Browser / All / 210 tasks` |
| Dataset SHA-256 | `b2e8626b1554decce4f0ca6b4aa463f24ec9667836c74b797666d34e2f6b90fe` |
| Agent | `browser-use` |
| Model | `gpt-5.5` |
| Judge | `gpt-5.4`, stepwise |
| Per-task limit | `40 steps / 600 seconds` |
| Browser arms | `lexmount`, `local` |

当前数据集为中文 118 条、英文 92 条，T1 194 条、T2 16 条，全部无需登录。

## 实验阶段

### 0. Preflight

- OpenAI `/models` 中存在 `gpt-5.5`，再用一次真实 completion 验证协议兼容性。
- 两套 Lexmount project 都通过 `sessions.list` 鉴权。
- 使用从固定 commit 创建的独立干净 worktree；运行脚本会将其忽略的 `config.yaml`
  指向本实验配置，不操作日常开发 checkout。
- `lexbrowser_eval.lexbench.probe_sessions` 验证指定数量的同时创建；若创建超时，会从异常中提取
  `session_id`，继续轮询并清理延迟激活的 session，再检查 active 数回到基线。
- 5090 主机确认 Chrome、代理/出口、systemd user scope、磁盘和至少 32 GiB Host 可用内存。

### 1. Paired smoke

使用 `task_sets/smoke.txt` 的 8 条任务，覆盖 zh/en 与 T1/T2。两端均以并发 2
执行并完成 Judge。检查配置快照脱敏、轨迹完整、session 清理、Chrome 清理和指标文件。

### 2. Paired pilot

使用 `task_sets/pilot20.txt`，两端依次以并发 5 执行。Pilot 用于确认模型限流、
本地出口、单条成本和 600 秒 timeout 是否合适，不用于发布最终成功率。

### 3. Full quality run

两端分别运行完整 210 条，固定并发 10。任务、模型、Agent 配置、Judge 和执行顺序
完全相同，只改变 `--browser lexmount|local`。基础设施失败可以重跑一次；正常 agent
失败和 timeout 不重跑。

### 4. Capacity and resource ladder

- 原始 session 容量：两套 project 分别执行指定并发探针；`--poll-timeout-seconds`
  只限制创建等待，`--cleanup-grace-seconds` 负责回收超时后才激活的 session。
- 端到端容量：`task_sets/capacity64.txt` 固定 64 条分层任务，在同一台 Linux runner
  反序比较 Lexmount 与 Local。当前正式矩阵为 c16/c32；只有 raw admission 真实观察到
  命名并发后才允许加入更高档位，避免把配置并发写成实际并发。
- 正式 c64 的当前 gate：实际任务构成 `28 EN + 36 ZH` 连续两次 raw admission 成功，
  或取得 provider reservation/profile allocation 证据。balanced `32 + 32` 只证明 quota
  64 可达，不替代该 gate。
- Local 使用 `MemoryMax=46G`，Host `MemAvailable < 32 GiB` 时触发护栏。
- 5090 不可用时允许在同一台 macOS 主机顺序运行两端；该 fallback 使用进程树 RSS，
  不把 RSS 写成 PSS，也不生成 cgroup/GPU 结论。
- `MACHINE_ID` 默认写成 `<platform>-<backend>`，移到命名主机时可以显式覆盖。macOS
  上 Lexmount arm 的资源数据只代表控制端进程树，不冒充远端服务端利用率。
- 一个点只有在计划任务全部形成轨迹、无 quota/OOM/护栏中止时才算可持续点。
- Lexmount 点还必须有 session monitor，且实际 active max 达到命名并发、monitor error 0、
  residual session 0。当前结果见
  [`results/gpt55-lexbench/overnight-20260713/`](../../results/gpt55-lexbench/overnight-20260713/README.md)。

### 4b. 5090 Local failure rerun

- `task_sets/local-linux-smoke12.txt` 先复测 12 条原始 Mac Local failure。
- `task_sets/local-mac-failures88.txt` 再覆盖全部 88 条原始 Mac Local failure。
- 敏感性分析保留原始 Local 的全部成功，并只把 5090 复测成功的 failure 加回 Local；
  这是刻意偏向 Local 的压力测试，不替代第二轮完整 210 条配对实验。
- 运行时记录两端出口 IPv4/ASN/地区是否一致；提交结果不公开公网 IP 明文。

### 5. Paired log and mechanism audit

- 用 `lexbrowser_eval.lexbench.audit_paired_runs` 联结 dataset、两端 summary、原始 `result.json` 和
  Judge NDJSON，审计所有单边成功任务。
- `task_sets/mechanism12.txt` 是按机制选择的诊断集，覆盖正向访问阻断、反向案例和
  阈值敏感案例；它只用于复现机制，不用于重新估计总体成功率。
- 两轮复测交换 backend 运行顺序，并用 `lexbrowser_eval.lexbench.summarize_replays` 将 synthetic
  Judge failure 标为 missing，而不是记为 0 分。

## 必须报告的指标

### 质量与稳定性

- planned / trajectory / judged / success 数量，分母分别展示
- Judge success rate 与配对差值
- 正常结束、timeout、session 创建失败、未捕获异常、导航/网络错误
- e2e 与 steps 的 mean、P50、P95
- token、模型成本、Judge 成本
- zh/en、T1/T2 分层结果

### 资源与容量

- cgroup CPU time 换算的平均 CPU cores
- 进程树 PSS mean/P95
- Chrome PSS mean/P95
- cgroup `memory.current` 峰值与 `memory.peak`
- Host MemAvailable 最小值
- 完成轨迹数 ÷ 有效 rollout 时长，单位 task/h
- GPU utilization、GPU memory、power 以及 `100% - avg utilization` 的 idle
- 如使用本地 vLLM：running/waiting queue 非零时间占比

GPT-5.5 由远端 API 提供，因此本轮 GPU 指标会采集但不用于解释两种浏览器后端的
因果差异；vLLM queue 指标标记为 N/A。CPU、内存、吞吐和稳定性仍与上一份报告同口径。

## 判定方式

- 首要结果是 210 条配对任务的成功率差值和 95% bootstrap CI，而不是两个独立百分比。
- 以 5 个百分点为实用非劣界值；同时单独展示两端基础设施失败率。
- 任何被 quota、OOM 或护栏截断的容量点都不能进入稳态资源对比表。
- 所有结论必须能回溯到固定 commit、dataset hash、配置快照和原始采样文件。

## 受控正控切片

`task_sets/controlled-parity-positive16.txt` 固定了 16 个曾在历史 GPT-5.5 阶段性运行中形成成功轨迹的任务。它只用于验证两个浏览器后端在可完成任务上的一致性，不用于估计 LexBench 总体分数。

`task_sets/stratified64-all.txt` 是从全部 210 条任务按 `language × task_type` 分层、使用固定
seed `20260719` 抽取的 64 条样本；`block-a` 与 `block-b` 是其不重叠的两半，分别采用
`local-first` 与 `lexmount-first` 执行。抽样器在
`scripts/generate_stratified_lexbench_taskset.py`，运行时应把生成的 manifest 与原始任务集一起归档。
该 API 的 GPT-5.5 端点拒绝 `temperature=0`，因此模型配置保持端点支持的 `1.0`；任务块交换执行顺序，避免把时间顺序误判为浏览器差异。

## 结果目录

```text
results/gpt55-lexbench/<run-id>/
  manifest.json
  resource_summary.json
  benchmark_summary.json
  followup.json
  report.md
```

原始轨迹、截图、日志和 `samples.csv` 默认不提交；需要共享时先做密钥与 URL 脱敏。

补测汇总可由保留在运行机的 artifact 重建：

```bash
uv run python -m lexbrowser_eval.lexbench.followup \
  --lexmount-full artifacts/gpt55-lexbench/full-lexmount-c10-20260712T171920Z/benchmark_summary.json \
  --local-full artifacts/gpt55-lexbench/full-local-c10-20260712T191703Z/benchmark_summary.json \
  --local-smoke artifacts/gpt55-lexbench/audit-local-c4-20260713T051950Z/benchmark_summary.json \
  --local-rerun artifacts/gpt55-lexbench/audit-local-c10-20260713T053447Z/benchmark_summary.json \
  --paired-audit results/gpt55-lexbench/20260713/paired-log-audit.json \
  --capacity-c10 artifacts/gpt55-lexbench/capacity-lexmount-c10-20260713T065014Z/benchmark_summary.json \
  --capacity-c64 artifacts/gpt55-lexbench/capacity-lexmount-c64-20260713T074319Z/benchmark_summary.json \
  --sessions-c64 artifacts/gpt55-lexbench/capacity-lexmount-c64-20260713T074319Z/session_samples.json \
  --probe-en-c64 artifacts/gpt55-lexbench/probe-en-c64-20260713T063020Z.json \
  --probe-zh-c64 artifacts/gpt55-lexbench/probe-zh-c64-20260713T063353Z.json \
  --output results/gpt55-lexbench/20260713/followup.json
```
