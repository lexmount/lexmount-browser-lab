# LexBench 高并发评测报告

## 结论

- **Lexmount Browser 最大可持续并发：20。** c40、c60 分别有 20、40 个实例在云端会话创建阶段触发 session quota。
- **Local Chrome 最大可持续并发：60。** c80 在持续内存压力下被 `systemd-oomd` SIGKILL；c100 未维持 60 秒且 PSS 采样失效，因此均不可持续。
- 同为 c20 时，Local 吞吐高 19.1%，但平均 CPU 约为 Lexmount 的 9.6 倍、进程树 PSS 约为 2.5 倍，并额外占用约 5.38 GiB Chrome PSS。
- Stage 2 使用官方 LexBench evaluator、`gpt-5.4`、stepwise 策略及官方阈值；147 条有效轨迹中 9 条通过，**已评分 Success 为 6.12%**。该分数衡量固定 20 任务在压力下的重复执行，不替代 All 210 正式质量分数。

## 实验口径

- 官方代码：`browseruse-agent-bench@ccd5fcbdfb975257b2ce38161dc9bc2ab294b420`，未修改官方源码。
- 固定随机种子 `20260710`，从 All 210 冻结 20 个任务；通过 1/3/5/10/25 个隔离 replica 提供 c20/c60/c100/c200/c500 负载。
- 每个 replica 使用官方 `bubench run`、`mode=specific`、`concurrency=20`；重复 task ID 通过独立 timestamp 隔离。
- Qwen 配置保持 `qwen3_8B`、max steps 40、task timeout 600 秒、structured JSON schema、flash mode。
- CPU 使用 cell cgroup 的累计 CPU time；内存使用 cgroup 内进程 PSS，Local Chrome 单独统计；Host 内存只用于安全护栏。
- Local评测cell整体限额：`TasksMax=32768`、`MemoryMax=46 GiB`，包括Agent、Chrome及其他子进程，并非单独给Chrome 46 GiB；同时保留至少32 GiB Host可用内存。c80的cgroup内存峰值为39.14 GiB，systemd-oomd在达到46 GiB硬上限前已因持续内存压力介入。
- 实习生持续使用 Qwen concurrency=1 是批准的背景负载。GPU 指标为包含该背景负载的整卡观察值，**不用于容量通过/失败判断**。

## 质量指标

`Judge Success` 只统计得到轨迹并被 gpt-5.4 评分的实例；`计划 E2E Success` 以计划实例为分母，包含 session quota、资源护栏和 systemd-oomd 导致的无结果实例。

| Backend | 目标并发 | 计划/已评分 | Success | Judge Success | 计划 E2E Success | Avg steps | Avg e2e(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lexmount | 20 | 20/20 | 2 | 10.00% | 10.00% | 12.30 | 418.73 |
| Lexmount | 40 | 40/20 | 2 | 10.00% | 5.00% | 12.50 | 303.56 |
| Lexmount | 60 | 60/20 | 2 | 10.00% | 3.33% | 9.70 | 471.56 |
| Local | 20 | 20/20 | 2 | 10.00% | 10.00% | 11.40 | 436.91 |
| Local | 60 | 60/60 | 1 | 1.67% | 1.67% | 7.30 | 505.85 |
| Local | 80 | 80/7 | 0 | 0.00% | 0.00% | 2.43 | 130.81 |
| Local | 100 | 100/0 | 0 | — | 0.00% | — | — |

总体：147 条已评分轨迹，9 条成功，Success 6.12%，Avg steps 9.34，Avg e2e 434.57 秒。

## 资源与容量指标

主表只保留通过容量判定的稳态点。CPU和内存均只统计评测cell的cgroup/进程树。

字段解释：

- **CPU cores**：rollout窗口内cgroup累计CPU时间除以wall time；2.39表示平均相当于持续占用2.39个CPU核。
- **PSS**（Proportional Set Size）：进程私有驻留内存全部计入，共享驻留页按共享进程数分摊；对Chrome这种多进程程序，比直接累加RSS更能避免共享内存重复计算。
- **PSS mean/P95**：每个采样点先汇总评测进程树PSS，再计算时间窗口平均值和第95百分位。P95表示95%的有效采样不超过该值，不是峰值，也不是平均值的95%。
- **Chrome PSS mean/P95**：总进程树PSS中仅Chrome/Chromium进程的子集。Lexmount浏览器运行在云端，因此5090侧该值为0。
- **cgroup内存峰值**：整个评测cell的最高内存记账值，还可能包含文件缓存和内核内存；它与PSS不是同一统计量。
- **≥95%持续(s)**：活跃Agent runner达到目标并发95%以上的最长连续时间，不等同于总rollout时长。
- **吞吐**：完整轨迹数除以有效rollout时长；只对通过容量判定的行进行横向比较。

| Backend | 并发 | 完成轨迹 | ≥95%持续(s) | CPU cores | PSS mean/P95 (GiB) | Chrome PSS mean/P95 (GiB) | cgroup内存峰值 (GiB) | 吞吐 (task/h) | 平均 GPU idle* | 平均 GPU SM* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexmount | 20 | 20/20 | 134.1 | 0.25 | 3.92 / 5.12 | 0 / 0 | 5.24 | 100.06 | 14.75% | 85.25% |
| Local | 20 | 20/20 | 60.6 | 2.39 | 9.64 / 11.22 | 5.38 / 6.45 | 12.29 | 119.21 | 3.12% | 96.88% |
| Local | 60 | 60/60 | 262.1 | 8.79 | 28.54 / 34.40 | 15.53 / 19.14 | 38.11 | **353.31** | 2.65% | 97.35% |

\* `平均 GPU idle = 100% - 平均 GPU SM utilization`。这是5090整卡的平均空闲比例，不是浏览器进程的GPU使用率；包含批准的外部Qwen concurrency=1及其他常驻进程，仅用于观察，不能用于容量判定或单独归因给浏览器后端。

### 失败尝试诊断

失败尝试不产生可比较的稳态吞吐、Success 或 GPU 指标，因此不放入主资源表。下列数据只说明失败发生的位置；其中资源值是失败前的部分窗口。

| Backend | 目标并发 | 完成轨迹 | ≥95%持续(s) | 失败原因 | 失败前诊断证据 |
|---|---:|---:|---:|---|---|
| Lexmount | 40 | 20/40 | 5.0 | session quota | 仅20个实例形成轨迹；其余在云端会话创建阶段失败 |
| Lexmount | 60 | 20/60 | 8.0 | session quota | 仅20个实例形成轨迹；其余在云端会话创建阶段失败 |
| Local | 80 | 7/80 | 149.0 | systemd-oomd | PSS P95 35.77 GiB、cgroup内存峰值39.14 GiB；随后SIGKILL 13179个进程 |
| Local | 100 | 0/100 | 25.0 | PSS采样失效，安全停止 | cgroup内存峰值39.67 GiB；PSS覆盖不完整，不报告稳态PSS |

`实际峰值` 原先统计的是活跃 Agent runner 进程，而不是成功创建的浏览器 session，已从报告移除。quota 行原先显示的吞吐只等于20条残留结果除以失败尝试时长，c80吞吐只覆盖被杀前的7条结果；两者均不代表该目标并发的有效吞吐。

### GPU 观察口径

同为c20时，Lexmount的整卡零利用率样本高于Local，原因是5090 GPU主要执行Qwen推理，而不是云端浏览器；浏览器交互等待改变了LLM请求到达节奏。

| c20观察指标 | Lexmount | Local |
|---|---:|---:|
| 平均 GPU idle | 14.75% | 3.12% |
| 平均 GPU SM | 85.25% | 96.88% |
| 完全零利用率的1秒样本 | 14.16% | 2.55% |
| Qwen running > 0 的时间占比 | 85.77% | 97.20% |
| Qwen waiting > 0 的时间占比 | 37.29% | 79.61% |

Local的Qwen running/waiting队列更连续，因此GPU更少出现空闲；Lexmount请求更脉冲化，存在更多浏览器/CDP/页面等待间隔。因此，本次c20条件下可以确认Lexmount rollout期间Qwen GPU空闲更多，但不能据此证明Lexmount端到端容量更高：它也可能表示远程浏览器等待没有持续喂满Qwen。当前Lexmount首先受session quota=20限制，只有提高云端session quota并重新测试c40/c60，才能验证这些GPU余量能否转化为更高吞吐或并发。

### 同并发 c20 对比

| 指标 | Lexmount | Local | Local 相对 Lexmount |
|---|---:|---:|---:|
| 吞吐 | 100.06 task/h | **119.21 task/h** | +19.1% |
| 平均 CPU | **0.25 cores** | 2.39 cores | 9.6× |
| 进程树 PSS | **3.92 GiB** | 9.64 GiB | +5.72 GiB |
| Local Chrome PSS | **0 GiB** | 5.38 GiB | +5.38 GiB |
| cgroup内存峰值 | **5.24 GiB** | 12.29 GiB | +7.05 GiB |
| Judge Success | 10.00% | 10.00% | 持平 |

## 容量与失败归因

| Backend | 最大可持续并发 | 首个不可持续点 | 容量失败原因 |
|---|---:|---:|---|
| Lexmount | **20** | 40 | 云端 session quota；c40 仅 20/40、c60 仅 20/60 产生轨迹 |
| Local | **60** | 80 | c80 在内存高水位持续受压后被 systemd-oomd SIGKILL，共终止 13179 个进程 |

Local c100 峰值达到目标，但只持续 25 秒，PSS 全量采样超过 15 秒后被监控器安全停止；根据预先约定，未启动 c200/c500。c80 原始 summary 曾被宽泛日志匹配误标为 `cdp_or_websocket`，实际 journal 已确认是 `systemd-oomd`；修正证据保存在 campaign 的 `diagnostic_corrections.json`，原始产物未覆盖。

### 模型、环境与无结果实例

| Backend | 计划 | 已评分 | Judge失败 | 无结果 | 模型类 M* | 环境类 E* | H* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lexmount | 120 | 60 | 54 | 60 | 45 | 6 | 3 |
| Local | 260 | 87 | 84 | 173 | 80 | 2 | 2 |

- **有结果但Judge错误：** 147 条轨迹中 138 条失败，主要是模型类；Lexmount 以 M1/M4 为主，Local 以 M1/M6 为主。
- **无结果：** Lexmount 的 60 条全部来自云端 session quota；Local 的 173 条来自 c100安全停止的100条和c80 systemd-oomd后的73条。
- **环境健康：** 正式执行中没有确认的Qwen连接失败或任务级timeout。c80的CDP宽泛匹配经journal复核不是根因。
- **质量趋势：** Local 从 c20 的10.00%下降到c60的1.67%，说明即使资源容量仍可承载60并发，模型输出质量在严重Qwen排队下明显下降。

## 可审计产物

- Campaign：`/data/wf/sxh/results/lexbench/lexbench_stress_20260710T151059Z/stress_process_attributed`
- Rollout summary：`rollout_summary.json`
- 诊断修正：`diagnostic_corrections.json`
- Stage 2：`stage2_gpt54_c5/`
- 聚合质量结果：`stage2_gpt54_c5/aggregate.json`
- 147 条实例级结果：`stage2_gpt54_c5/stress_gpt-5.4_stepwise_results.jsonl`

所有 Stage 2 结果均位于隔离目录；源轨迹的 hash/mtime 未变化，源 run 未写入 `tasks_eval_result`。
