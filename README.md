# Lexmount Browser Lab — H100 GRPO 训练复现

本仓库只做一件事:在 H100 上复现我们已验证的 Browser-RL 训练——

**Qwen3-8B + verl GRPO + Lexmount 云浏览器(NeMo-Gym 环境服务)+ WebVoyager 真实网站任务 + LLM 裁判二元奖励,60 步,reward 均值 ~0.105 → ~0.289。**

全部所需内容都在 [`training/h100/`](training/h100/) 一个目录里:启动脚本、训练运行时、任务数据,自包含。

## 快速开始

1. 准备:一台 8×H100-80GB(或两台)、Docker + NVIDIA container toolkit、两个 API key(Lexmount 浏览器 + deepseek-v4-flash 裁判接口)。
2. 照着 **[training/h100/README.md](training/h100/README.md)** 走六步——每步一条命令,直到看见 `LAUNCH_OK`。
3. 曲线看 TensorBoard 的 `critic/rewards/mean`,参考数字与预期形状同在该 README。

## 目录

| 路径 | 内容 |
|---|---|
| `training/h100/README.md` | 复现指南(准备 → 六步启动 → 看结果) |
| `training/h100/` 其余 | 启动脚本、`runtime/` 训练运行时、`data/` 任务数据(含哈希链) |
