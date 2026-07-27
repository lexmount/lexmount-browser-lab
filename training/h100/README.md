# H100 GRPO 训练复现指南

本目录是一套**自包含**的训练复现包:在 H100 上复现我们已验证的 Browser-RL 训练配方,得到同样形状的 reward 增长曲线。

一句话说明这套配方是什么:**Qwen3-8B 模型,用 verl GRPO 训练,浏览器环境是 Lexmount 云浏览器(经 NeMo-Gym 环境服务接入),任务是 WebVoyager 真实网站任务,奖励由 LLM 裁判(deepseek-v4-flash)二元判分。**每个训练步采样 8 个任务 × 每任务 8 条轨迹 = 64 条真实网页操作轨迹,判分后做一次 GRPO 更新,共 60 步。

该配方经内部完整验证(2026-07-21,16 卡,60 步):reward 均值从前 10 步的 ~0.105 爬升到后 10 步的 ~0.289。**本目录所有超参数与验证运行完全一致**,硬件层按 H100/CUDA 适配。

---

## 开始之前:你需要准备什么

| 项目 | 要求 |
| --- | --- |
| 硬件 | 1 台 8×H100-80GB(默认);或 2 台同规格(head 到 worker 需免密 SSH)。每次训练预留 ~300 GB 磁盘(单个 checkpoint 约 92 GB,60 步存 3 个) |
| 软件 | Docker + NVIDIA container toolkit;能访问 GitHub / PyPI / Docker Hub / Hugging Face / Lexmount API,以及任务集里的公开网站(arxiv.org、bbc.com、coursera.org、github.com) |
| 凭证 | ① Lexmount API key + project ID(浏览器会话;训练稳态用 64 并发,账号配额建议 ≥ 80 留出关闭确认/清理余量,见「会话生命周期与配额」);② 一个能调用 `deepseek-v4-flash` 的 OpenAI 兼容接口(裁判用,任何服务商或自部署均可)。逐项说明见 `secrets.env.example` |
| 模型 | Hugging Face 上的官方 `Qwen/Qwen3-8B` 权重 |

---

## 六步启动训练

所有命令都在 head 节点、仓库根目录下执行。

### 第 1 步:克隆代码

```bash
git clone https://github.com/lexmount/lexmount-browser-lab.git && cd lexmount-browser-lab
```

复现所需的一切都在 `training/h100/` 子树里:启动脚本、训练运行时(`runtime/`)、任务数据(`data/`)。168 题任务清单已随仓库提供,训练用的 parquet 会在首次启动时自动生成(每一步数据推导都有 SHA256 校验,见 `data/webvoyager-clean/MANIFEST.json`)。

### 第 2 步:构建训练镜像

```bash
docker build -f training/h100/Dockerfile -t lexbrowser-verl-h100:local training/h100
```

镜像在 `vllm/vllm-openai:v0.18.0` 之上叠加钉死版本的 verl 和环境包。双节点时两台机器都要有这个镜像(可 `docker save`/`docker load` 分发)。

### 第 3 步:下载策略模型

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B
```

放哪都行,启动时用 `MODEL_PATH` 传入;双节点需两台机器同路径。

### 第 4 步:安装浏览器环境服务(每台 head 一次)

```bash
bash training/h100/install_nemo_gym_runtime_h100.sh
```

自动克隆 NVIDIA NeMo-Gym v0.2.1(带 commit 校验)并安装它的几个 CPU 依赖,默认装到 `/data/lexbrowser-rl`。

### 第 5 步:填写凭证

```bash
cp training/h100/secrets.env.example training/h100/secrets.env
chmod 600 training/h100/secrets.env
$EDITOR training/h100/secrets.env    # 填 Lexmount 和裁判接口的 key,每个变量文件里有注释
```

### 第 6 步:启动

单节点(8×H100):

```bash
NODES_CSV=<本机IP> MODEL_PATH=/models/Qwen3-8B bash training/h100/launch_h100.sh
```

双节点(16×H100,与验证运行同等规模):

```bash
NODES_CSV=<head-IP>,<worker-IP> MODEL_PATH=/models/Qwen3-8B bash training/h100/launch_h100.sh
```

启动器会依次:生成数据 parquet → 逐节点预检(GPU、磁盘、镜像、CUDA,外加一次**真实的** Lexmount 会话+CDP+网页抓取探测,凭证有问题在这一步就会报出来)→ 启动浏览器环境服务和 Ray → 等所有卡注册 → 提交 60 步训练。看到 `LAUNCH_OK` 即启动成功。

**默认值就是验证配置,以复现曲线为目标时什么都不要改。**可选开关:`SKIP_PREFLIGHT=1` 跳过预检;`STAMP=<run-id>` 指定运行名;`RESUME_FROM_PATH=...` 断点续跑;`PPO_MAX_TOKEN_LEN_PER_GPU=12288` 使用内部验证时的原始打包预算(默认 15360 为 80GB 显存等比放大值,只影响吞吐不影响训练语义)。

---

## 怎么看结果

```bash
tail -f  <RUNS_ROOT>/<run-id>/logs/train.log                # 训练日志
tensorboard --logdir <RUNS_ROOT>/<run-id>/tensorboard       # 曲线
docker logs -f lexbrowser-nemo-gym-webvoyager               # 浏览器环境服务
```

- 曲线看 TensorBoard 里的 **`critic/rewards/mean`**。
- 每步的轨迹明细在 `<run-dir>/rollouts/`(JSONL,分数字段名为 `score`);裁判的输入输出审计在 `<run-dir>/audit/judge_io.jsonl`。
- 训练结束会自动运行 `verify_rollout_groups.py`,校验 8×8 分组全程无破损。

**参考曲线**(验证运行,Lexmount 后端,60 步 3840 条轨迹):

| 指标 | 数值 |
| --- | --- |
| reward 均值,第 1–10 步 | 0.105 |
| reward 均值,第 11–50 步 | 0.280 |
| reward 均值,第 51–60 步 | 0.289 |
| 全程均值 | 0.252 |
| 正奖励轨迹 | 969 / 3840 |
| 全零奖励步 | 0 / 60 |

预期形状:前几步在 ~0.1 附近,约 15–20 步后爬到 ~0.25–0.30 平台,单步波动大(64 条里正例 1~40 条都出现过)——**看趋势,不看单点**。本地 Chromium 对照组呈相同形态(0.145→0.272),说明增长来自配方而非某个后端。两次验证运行的原始 TensorBoard 归档可提供比对。

**耗时预期**:内部验证运行 16 卡约 4 小时(238 秒/步,其中 ~87% 是浏览器/裁判等待而非 GPU 计算)。16×H100 应同级或更快;单节点 8 卡时每波并发轨迹减半,预计 6–8 小时——曲线按步数索引,单节点复现同样的曲线,只是慢一些。

**两个不要慌的现象**:
- 某一步 64 条轨迹全 0 分(或全 1 分)时,GRPO 组内优势为零,该步 `grad_norm=0`——这是零方差步的正常行为,不是故障;基座模型在最初几步常见。
- rollout 结束阶段偶发个别 Lexmount 会话清理超时告警,不影响轨迹与判分。

---

## 会话生命周期与配额(2026-07-27 加固)

针对一次真实复现事故(step 61–80 大面积 `ERROR_ENVIRONMENT_BROWSER_RESET`,根因是环境服务在训练侧放弃后仍注册会话、以及超时重试放大了 provider 侧并发)的加固。默认值开箱即用,无需改动;这里说明背后的约束,便于调参时不破坏它们。

**硬性不变量:provider 侧可见的会话数任意时刻 ≤ `LEXMOUNT_MAX_CONCURRENT_SESSIONS`(默认 64)。** 并发槽位跟随会话在 provider 侧的真实生命周期:create 请求发出即占位,超时被放弃的 create 仍占位直到迟到会话被关闭,close 未确认前不还位。账号配额应高于该值留出余量(建议配额 ≥ 会话数 + 16)。

| 环境变量 | 默认 | 约束 |
| --- | --- | --- |
| `LEXMOUNT_MAX_CONCURRENT_SESSIONS` | 64 | = 每步轨迹数(8 任务 × 8 采样);须低于账号配额并留余量 |
| `LEXMOUNT_SESSION_CREATE_ATTEMPTS` | 2 | 每次 /reset 内的 create 尝试数(带指数退避;配额类错误等更久) |
| `LEXMOUNT_RESET_TOTAL_BUDGET_S` | 110 | 单次 /reset 的服务端总墙钟(含重试);**必须小于** 训练侧 `LEXMOUNT_RESET_REQUEST_TIMEOUT_S`(默认 120),否则训练侧放弃后服务端会注册无人认领的孤儿会话 |
| `LEXMOUNT_RESET_CIRCUIT_COOLDOWN_S` | 60 | reset 成功率熔断(近 32 次 < 20% 即快速失败)的冷却时间 |
| `LEXMOUNT_SWEEP_INTERVAL_S` | 60 | 孤儿会话对账周期:连续两轮出现在 provider 侧(`status="active"`)、但本服务不认识的会话会被删除。**要求账号专用于本次训练**;共享账号请设为 0 关闭 |
| `LEXBROWSER_JUDGE_REQUEST_TIMEOUT_S` | 45 | 裁判单次调用超时(裁判在会话释放之后运行,不占浏览器) |
| `LEXBROWSER_ENV_FAILURE_RETRIES` | 1 | 环境失败 rollout 的整集重采样次数(新会话重跑该 episode) |
| `LEXBROWSER_GRPO_MIN_VALID_PER_GROUP` | 2 | GRPO 组内有效样本低于此数时整组优势置 0(基线不再有意义) |

**环境失败与策略失败的区分**(三层):① reset 失败、/close 传输失败、会话丢失、裁判不可用等基础设施故障,先按 `LEXBROWSER_ENV_FAILURE_RETRIES` 用新会话**重采样整条 episode**(失败路径有预算/熔断兜底,快速失败,重试代价低);② 重试后仍无效的样本标记 `lexbrowser_invalid_sample` 并全 response loss-mask;③ verl 补丁 `runtime/patches/core_algos.py` 让 GRPO **组均值/方差只在有效样本上计算**——无效样本优势恒 0、不进组统计,组内有效样本不足阈值时整组优势置 0。环境故障因此既不产生策略梯度,也不再压低组基线、放大幸存样本的优势。策略自身的错误(不合法动作、选择器失效、重复无进展)不受影响,照常参与训练。`/close` 服务端幂等(结果缓存可重放),训练侧超时重试能取回真实判分。`/health` 暴露 `live_sessions`、`unconfirmed_closes`、`leaked_sessions`、`sweeper_reclaimed`、`reset_circuit_open` 等仪表,复现时建议接入监控。

## 附录

### 版本与出处(Provenance)

| 项目 | 值 |
| --- | --- |
| 验证运行 | 2026-07-21 内部双后端对照(Lexmount 云浏览器 vs 本地 Chromium),各 60 步 |
| verl | 0.9.0.dev0,git commit `30119a253087bff86c12d329d2d8dd43c589705f` |
| vLLM | 0.18.0 |
| torch | 2.10.0+cu(由 vLLM 0.18.0 wheel 钉定) |
| transformers | 5.3.0 |
| NeMo-Gym | v0.2.1,commit `27e921137042dcdb8a39c7169128619b9108074b` |
| 策略模型 | Qwen/Qwen3-8B(HF 官方权重) |
| 裁判模型 | `deepseek-v4-flash`(任意 OpenAI 兼容接口) |
| 训练数据 | webvoyager-clean 168 题,哈希链见 `data/webvoyager-clean/MANIFEST.json` |

### 架构

```text
verl GRPO(8 或 16 张 H100)
  -> verl 异步 vLLM rollout(TP=4,hermes 多轮工具调用)
  -> agent loop(runtime/lexbrowser_verl_agent.py)
  -> HTTP -> NeMo-Gym 环境服务(纯 CPU,runtime/nemo_gym_webvoyager_server.py)
       -> WebVoyager 环境(runtime/lexbrowser_webvoyager/)
       -> Lexmount 云浏览器(CDP)
       -> OpenAI 兼容裁判(deepseek-v4-flash)
  -> 二元奖励 -> GRPO 优势 -> FSDP 更新
```

### 文件一览

| 文件 | 作用 |
| --- | --- |
| `launch_h100.sh` | 一条命令编排:预检 → 环境服务 → Ray → 训练 |
| `run_lexbrowser_grpo_h100.sh` | verl 训练入口(验证超参即默认值) |
| `start_ray_node_h100.sh` | 每节点 Ray 容器(head/worker) |
| `start_nemo_gym_webvoyager_server_h100.sh` | CPU 浏览器环境服务 |
| `install_nemo_gym_runtime_h100.sh` | NeMo-Gym v0.2.1 一次性安装 |
| `preflight_h100.sh` | 逐节点预检(含真实 Lexmount 浏览器探测) |
| `build_webvoyager_clean_data.py` | 重建 168 题训练集(哈希校验) |
| `Dockerfile` / `requirements-cuda.txt` | 钉版本的 CUDA 软件栈 |
| `secrets.env.example` | 全部所需凭证,逐项注释 |
| `data/webvoyager-clean/` | 168 题清单 + 哈希链 |
| `runtime/` | 与验证运行一致的训练运行时(agent loop、环境服务、补丁、数据转换) |
