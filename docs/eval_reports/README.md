# 评估报告索引

本目录只保存最终评估报告，不保存巡检记录、临时日志、原始轨迹或内部实施计划。

## 已完成评估

| Benchmark | 模型 | 任务规模 | Lexmount | Local | 报告 |
|---|---|---:|---:|---:|---|
| LexBench-Browser | Qwen3-8B | All 210 | 11.43% | 8.57% | [质量与资源汇总](lexbench/README.md) |
| LexBench压力测试 | Qwen3-8B | 固定20任务重复实例 | 最大可持续并发20 | 最大可持续并发60 | [压力与容量](lexbench/stress_results.md) |
| Online-Mind2Web | Qwen3-8B | 300 | 5.00% | 5.67% | [最终汇总](online-mind2web/README.md) |

GPT-5.5 LexBench结果沿用main的可审计产物布局，位于
[`results/gpt55-lexbench/20260713/report.md`](../../results/gpt55-lexbench/20260713/report.md)。

WebArena-Lite当前只有官方Playwright runner，尚无可发布的双后端评估结果。

## 资源口径

- CPU：评测cgroup或进程树消耗的平均CPU cores，不使用Host-wide CPU作效率结论。
- Linux内存：进程树PSS及其中的Chrome PSS；cgroup内存峰值只作为容量和安全指标。
- macOS内存：进程树RSS；不与Linux PSS混算。
- GPU：共享Qwen vLLM的整卡观察值，不属于浏览器进程树；存在外部消费者时不用于容量判定。
- 吞吐：通过容量判定的完整rollout完成轨迹数除以有效rollout时间。

## 详细报告

### LexBench-Browser / Qwen3-8B

- [两后端汇总](lexbench/README.md)
- [Lexmount正式结果](lexbench/lexbrowser_results.md)
- [Local正式结果](lexbench/local_results.md)
- [压力、资源与容量报告](lexbench/stress_results.md)

### Online-Mind2Web / Qwen3-8B

- [两后端汇总](online-mind2web/README.md)
- [Lexmount正式结果](online-mind2web/lexbrowser_results.md)
- [Local正式结果](online-mind2web/local_results.md)
