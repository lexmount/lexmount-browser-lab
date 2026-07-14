# LexBrowser WebVoyager 训练报告

正式 100-step 训练尚未完成，因此当前不存在可诚实展示的 reward 曲线。

已完成的真实 WebVoyager smoke 证据：12,288-token packing、TP=2、Qwen3-1.7B LoRA、
`1 prompt × 8 rollouts × 1 optimizer update` 可以完成 rollout、logprob 和 backward/update，
但当前 Lexmount 项目的 Web egress 对关键训练站点不稳定：Google 返回
`ERR_TUNNEL_CONNECTION_FAILED`，Allrecipes 多数会话落入 Cloudflare challenge，WolframAlpha
当前页面不再暴露任务要求的数值结果。它们均已被标注为基础设施失败或 judge 的真实拒绝，
没有被伪装成 reward 改善。

环境支持通过 `LEXMOUNT_EXTERNAL_PROXY_SERVER`、
`LEXMOUNT_EXTERNAL_PROXY_USERNAME`、`LEXMOUNT_EXTERNAL_PROXY_PASSWORD` 配置 Lexmount 原生的
authenticated external proxy。获得可达 egress 后重新运行训练，
`training/scripts/generate_train_report.py` 会从 TensorBoard event 自动替换本页，并生成：

- `reward_vs_step.png`
- `browser_behavior_vs_step.png`
- `optimization_vs_step.png`
- `training_scalars.csv`
- `summary.json`

报告只使用实际训练日志，不预设 reward 一定上升。
