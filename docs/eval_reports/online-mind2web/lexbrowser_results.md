# Online-Mind2Web 评测报告：Qwen3-8B + Lexmount Browser

## 结论摘要

本轮使用固定的 Online-Mind2Web 300 tasks、Qwen3-8B、Lexmount Browser 和 rollout concurrency=10。仅运行官方 `WebJudge_Online_Mind2Web_eval` 分支；Judge 为 gpt-5.4、temperature=1、concurrency=10。

最终成功率为 **5.00%（15/300）**。

## 正式指标

| 指标 | 结果 | 口径 |
|---|---:|---|
| Success Rate | **5.00%（15/300）** | 全部300项为分母 |
| Judge失败 | 280 | 279条官方失败 + 1条内容策略强制失败 |
| Rollout强制失败 | 5 | 用户要求立即收口，未形成严格有效轨迹 |
| 总失败 | 285 | `300 - 15` |
| Avg steps | 9.29 | 295条严格有效轨迹 |
| Avg e2e | 128.34 s | 295条严格有效轨迹 |

## 覆盖与审计

| 项目 | 数量 |
|---|---:|
| 计划任务 | 300 |
| 严格 `online-mind2web-v2` 轨迹 | 295 |
| 官方 WebJudge JSONL 唯一记录 | 294 |
| 内容策略强制失败 | 1 |
| Rollout强制失败 | 5 |

内容策略失败任务：

```text
4d3157aab34b54e5f0c4b965dfe930f3
```

因最终时限按用户授权直接计失败的任务：

```text
c1d6ea6f2196d25782cc3646ff3090db
29b7372d5a3884a2ba831af2d117af3c
b99c02965196d51e80ac7539e33f335b
ba01ea557b73f864c35ebba0dd6f3cb2
662ae0f2d3ac851dbcdd245f908277e3
```

以上5项均因缺少精确 step screenshot 未通过严格轨迹门禁。它们没有伪造 Judge 输出，而是直接进入300项总分母并按失败计数。

## 固定配置

- Campaign：`20260711_001343`
- Dataset：`osunlp/Online-Mind2Web@84038480c979f3744ffadac18883b7095f90b332`
- OSU evaluator：`f0d805ee0e9e0b3ea70911e45e5264b72968f3dc`
- Bubench：`ccd5fcbdfb975257b2ce38161dc9bc2ab294b420`
- Agent：Qwen3-8B，`max_steps=40`，task timeout=600秒
- Browser：Lexmount
- Rollout concurrency：10
- Judge：gpt-5.4，temperature=1，max_tokens=512，3次重试，concurrency=10

## 与官方流程的关系

任务级 Judge 使用官方 prompt、关键点识别、关键截图识别、最终判定和 `predicted_label` 解析。明确偏差为 Judge backbone、temperature、并发，以及本次时限下5项未完成轨迹直接计失败。该结果不是 OSU Leaderboard 官方验证成绩。

## 远端产物

```text
/data/wf/sxh/results/online_mind2web_v2/20260711_001343/lexmount/official_webjudge/WebJudge_Online_Mind2Web_eval_gpt-5.4_score_threshold_3_auto_eval_results.json
/data/wf/sxh/results/online_mind2web_v2/20260711_001343/lexmount/official_webjudge/forced_failures.json
/data/wf/sxh/results/online_mind2web_v2/20260711_001343/lexmount/rollout_forced_failures.json
```
