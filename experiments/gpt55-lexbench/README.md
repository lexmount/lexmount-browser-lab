# GPT-5.5 × LexBench Browser Stability

实验来源：

- [0709 Lexmount Browser 接入 Nemo-gym 实验计划](https://lexmount.notion.site/0709-lexmount-browser-nemo-gym)
- [上一轮 Lexmount Browser 与 Local Chrome 评测结果](https://app.notion.com/p/399e7e998fb28085b947c3efaa7859f3)

## 固定条件

| 项目 | 值 |
| --- | --- |
| Benchmark runner | `lexmount/browseruse-agent-bench@b9d3dc655aec3cd10fd1fb86cfb7678bb9a27399` |
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
- `probe_lexmount_sessions.py` 逐级验证 20、40、60 个同时存活 session，并在每级结束后确认清理完成。
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

- 原始 session 容量：Lexmount 20 → 40 → 60，失败后停止上探。
- 端到端容量：两端先比较 c20，再以 20 为步长上探。
- Local 使用 `MemoryMax=46G`，Host `MemAvailable < 32 GiB` 时触发护栏。
- 一个点只有在计划任务全部形成轨迹、无 quota/OOM/护栏中止时才算可持续点。

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

## 结果目录

```text
results/gpt55-lexbench/<run-id>/
  manifest.json
  resource_summary.json
  benchmark_summary.json
  report.md
```

原始轨迹、截图、日志和 `samples.csv` 默认不提交；需要共享时先做密钥与 URL 脱敏。
