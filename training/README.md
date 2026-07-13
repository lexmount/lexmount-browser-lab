# LexBrowser WebVoyager GRPO

这套配置把 BrowserEnv 的 `browserbase/webvoyager-no-anti-bot` 迁移到
`lexbrowser/webvoyager-no-anti-bot`。训练框架改为 NVIDIA NeMo RL，浏览器实例由
Lexmount Browser 创建；任务、DOM 工具语义、600 条数据和二值 LLM judge 契约保持一致。

## 已固定的实验配置

| 项目 | 配置 |
| --- | --- |
| 基座模型 | `/home/wf/models/Qwen3-1.7B` |
| 算法 | 同步 multi-turn GRPO；NeMo Gym 持有完整浏览器轨迹 |
| 参数更新 | 2 GPU FSDP2 + DTensor v2 LoRA（rank 8，alpha 32，all-linear） |
| 数据 | Browserbase 环境包 0.1.4 中的 600 条过滤后 WebVoyager 数据 |
| 数据校验 | 600 个唯一 ID；源 JSONL SHA256 为 `b901adc3f1fb93c069260e1940c59b214374f0ffe58ff7dcf5b1af831d3b1097` |
| epoch / step | 最多 2 epoch；固定 100 optimizer steps（第 76 step 起进入第二次 shuffled pass） |
| 每 step | 8 prompts × 每 prompt 8 rollouts = 64 trajectories |
| 每条轨迹 | 最多 100 个环境 turn；每次模型调用最多生成 1024 tokens |
| Qwen 工具解析 | thinking 开启；Hermes tool parser；DeepSeek-R1 reasoning parser |
| reward | 有 tool call 后由 `glm-5.2` 根据任务和工具轨迹判定 yes/no；无 tool call 直接为 0 |
| 浏览器 | Lexmount cloud Chrome/CDP → 本地 Stagehand v3 DOM session |
| 浏览器并发 | 最多 20 个活跃 Lexmount session；其余 rollout 在 agent 侧排队 |
| 精度/内存 | bf16、activation checkpointing、LoRA、8K training packs；Stagehand/vLLM 32K context；vLLM 显存比例 0.65 |
| 输出 | `logs/lexbrowser-grpo/`、`results/lexbrowser-grpo/` |

`grpo.max_rollout_turns=1` 不表示浏览器只有一步。一次 NeMo Gym 请求内部会执行最多
100 个 `navigate/observe/act/extract` turn，并把完整 trajectory 作为一条 RL rollout 返回。

## 一键运行

服务器工作目录：

```bash
cd /data/wf/sxh/workspace/LexBrowserEnv
```

先验证完整链路（1 个任务、2 条 rollout、1 个 optimizer step）：

```bash
./run_lexbrowser_training.sh smoke
```

正式训练：

```bash
./run_lexbrowser_training.sh
```

脚本会自动执行数据转换、GPU/密钥/权限预检、拉取固定的 NeMo RL 0.6.0 NGC
镜像、启动双卡容器并保存日志。当前服务器用户不在 Docker group，首次执行会请求
`sudo` 密码。

## 文件职责

- `configs/grpo_lexbrowser_webvoyager_qwen3_1_7b_2x5090.yaml`：NeMo RL/GRPO/双卡配置。
- `lexbrowser_webvoyager/`：可被 Verifiers 加载的环境包及原始 600 条数据。
- `nemo_gym/lexbrowser_webvoyager.yaml`：NeMo Gym agent 实例配置。
- `nemo_gym/verifiers_agent_app.py`：保留 tool call、tool result、token id 和 logprob 的 NeMo Gym agent 兼容层。
- `nemo_rl_patches/vllm_worker.py`：给本地 Qwen 增加仅用于 API 的 `gpt-4o` alias，使 Stagehand v3 走 Chat Completions；实际权重始终是 Qwen3-1.7B。
- `nemo_rl_patches/vllm_worker_async.py`：只对带 token-prefix 的 policy rollout 强制 on-policy sampling；允许环境侧 Stagehand DOM planner 使用自己的 temperature。
- `scripts/prepare_webvoyager_data.py`：确定性生成 NeMo Gym JSONL，并检查数据行数和 ID。
- `scripts/run_lexbrowser_grpo.sh`：容器启动、挂载、预检和清理逻辑。

## 凭证

真实凭证只能放在仓库根目录的 `secrets.env`，该文件已被 Git 忽略且启动脚本要求
权限为 `600`：

```bash
cp secrets.env.example secrets.env
chmod 600 secrets.env
```

不要把凭证写进 YAML、日志或命令行参数。

## “Lexmount 不弱于 Browserbase”如何验收

训练能跑通只证明系统集成有效，不足以证明后端不弱。最终 A/B 必须固定模型 checkpoint、
采样参数、task 顺序、judge 和重试规则，只切换 browser provisioner，然后报告：

1. 任务成功率及 bootstrap 95% CI；最好使用 paired task-level 差值。
2. 环境失败率（建连、CDP、页面加载、tool timeout）和失败类型。
3. 每条成功轨迹的端到端时延、浏览器分钟数及成本。
4. 20 并发下的成功率和吞吐；再逐级升并发寻找退化点。

只有当 Lexmount 的成功率差值下界高于预先设定的非劣效界值（例如 -2 个百分点），
才能严谨地称为“不弱于”。600 条 WebVoyager 被用于训练时不能再当作独立泛化测试集；
后端工程 A/B 可以在相同任务上做 paired replay，但模型能力评测应另留未训练任务。
