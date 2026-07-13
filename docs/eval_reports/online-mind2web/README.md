# Online-Mind2Web：Lexmount Browser 与 Local Chrome 对比

本轮评测对象均为 Qwen3-8B、Online-Mind2Web 300 tasks、rollout concurrency=10。两端只运行官方 `WebJudge_Online_Mind2Web_eval` 分支，Judge 使用 gpt-5.4、temperature=1、concurrency=10。

## 最终结果

| 指标 | Lexmount Browser | Local Chrome | Local 相对 Lexmount |
|---|---:|---:|---:|
| Success Rate | **5.00%（15/300）** | **5.67%（17/300）** | **+0.67 pp** |
| 总失败 | 285 | 283 | -2 |
| Judge内容策略失败 | 1 | 1 | 0 |
| Rollout强制失败 | 5 | 1 | -4 |
| 严格有效轨迹 | 295/300 | 299/300 | +4 |
| Avg steps（有效轨迹） | 9.29 | 9.14 | -0.15 |
| Avg e2e（有效轨迹） | **128.34 s** | 158.60 s | +30.26 s |

Local Chrome 的最终成功数比 Lexmount 多2项；但本轮在用户要求立即收口时，Lexmount 尚有5项、Local尚有1项未形成严格有效轨迹，均按失败计入300项分母。因此质量差异需要结合不同的强制失败数量解读。

## 评测口径

- Dataset revision：`84038480c979f3744ffadac18883b7095f90b332`
- Agent：qwen3-8B
- Browser：Lexmount / 5090 Local Chrome
- Rollout concurrency：10
- Judge：仅 `WebJudge_Online_Mind2Web_eval`
- Judge backbone：gpt-5.4
- Judge temperature：1
- Judge concurrency：10
- Judge max_tokens：512
- Judge retries：3
- 指标：`Success Rate = 成功任务数 / 300 × 100%`

## 覆盖处理

官方 WebJudge JSONL 保持原始、可审计，不写入伪造记录。经官方重试仍被内容策略拒绝的任务记录在各自 `forced_failures.json` 中并计失败；未通过严格轨迹门禁且按用户最终授权停止重跑的任务记录在 `rollout_forced_failures.json` 中并计失败。

详细报告：

- [Lexmount Browser](lexbrowser_results.md)
- [Local Chrome](local_results.md)

> 该结果采用官方 WebJudge prompt与判定逻辑，但 Judge backbone、temperature、并发和最终强制失败策略属于本轮明确偏差；不是 OSU Leaderboard 官方验证成绩。
