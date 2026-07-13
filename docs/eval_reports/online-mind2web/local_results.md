# Online-Mind2Web 评测报告：Qwen3-8B + Local Chrome

## 结论摘要

本轮使用固定的 Online-Mind2Web 300 tasks、Qwen3-8B、5090本地 Chrome/Xvfb 和 rollout concurrency=10。仅运行官方 `WebJudge_Online_Mind2Web_eval` 分支；Judge 为 gpt-5.4、temperature=1、concurrency=10。

最终成功率为 **5.67%（17/300）**。

## 正式指标

| 指标 | 结果 | 口径 |
|---|---:|---|
| Success Rate | **5.67%（17/300）** | 全部300项为分母 |
| Judge失败 | 282 | 281条官方失败 + 1条内容策略强制失败 |
| Rollout强制失败 | 1 | 用户要求立即收口，未形成严格有效轨迹 |
| 总失败 | 283 | `300 - 17` |
| Avg steps | 9.14 | 299条严格有效轨迹 |
| Avg e2e | 158.60 s | 299条严格有效轨迹 |

## 覆盖与审计

| 项目 | 数量 |
|---|---:|
| 计划任务 | 300 |
| 严格 `online-mind2web-v2` 轨迹 | 299 |
| 官方 WebJudge JSONL 唯一记录 | 298 |
| 内容策略强制失败 | 1 |
| Rollout强制失败 | 1 |

内容策略失败任务：

```text
2218042362d8fae73756eb309848c2b2
```

因最终时限按用户授权直接计失败的任务：

```text
c521933dad9c0ef9f1dfa2f38b8e4405
```

该项没有伪造 Judge 输出，而是直接进入300项总分母并按失败计数。

## 固定配置

- Campaign：`20260711_001343`
- Dataset：`osunlp/Online-Mind2Web@84038480c979f3744ffadac18883b7095f90b332`
- OSU evaluator：`f0d805ee0e9e0b3ea70911e45e5264b72968f3dc`
- Bubench：`ccd5fcbdfb975257b2ce38161dc9bc2ab294b420`
- Agent：Qwen3-8B，`max_steps=40`，task timeout=600秒
- Browser：5090 Local Chrome/Xvfb
- Rollout concurrency：10
- Judge：gpt-5.4，temperature=1，max_tokens=512，3次重试，concurrency=10

## 与官方流程的关系

任务级 Judge 使用官方 prompt、关键点识别、关键截图识别、最终判定和 `predicted_label` 解析。明确偏差为 Judge backbone、temperature、并发，以及本次时限下1项未完成轨迹直接计失败。该结果不是 OSU Leaderboard 官方验证成绩。

## 远端产物

```text
/data/wf/sxh/results/online_mind2web_v2/20260711_001343/local/official_webjudge/WebJudge_Online_Mind2Web_eval_gpt-5.4_score_threshold_3_auto_eval_results.json
/data/wf/sxh/results/online_mind2web_v2/20260711_001343/local/official_webjudge/forced_failures.json
/data/wf/sxh/results/online_mind2web_v2/20260711_001343/local/rollout_forced_failures.json
```
