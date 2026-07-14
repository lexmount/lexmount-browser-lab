# LexBrowser WebVoyager 训练启动

在训练服务器执行：

```bash
cd /data/wf/sxh/workspace/LexBrowserEnv
```

## 1. 首次准备

确认已配置凭证，且文件不会被提交：

```bash
cp secrets.env.example secrets.env   # 仅首次、尚未创建时执行
chmod 600 secrets.env
nvidia-smi                           # 需要两张 GPU；启动前应有至少 22 GiB 空闲显存/GPU
```

`secrets.env` 必须包含 Lexmount 与 judge 所需的变量；不要将其提交到 Git。

## 2. 先运行两道真实网页门禁

门禁均使用 Lexmount、真实 WebVoyager 网站和真实 GRPO update，但不会写正式 checkpoint。

```bash
# 1 task × 8 rollout，micro batch = 1
./run_lexbrowser_training.sh stage1

# 1 task × 8 rollout，micro batch = 4
./run_lexbrowser_training.sh stage2
```

两项都以 `Loss:` 和 `Avg Reward:` 输出、且进程正常退出后，再启动正式训练。

## 3. 启动正式训练

```bash
./run_lexbrowser_training.sh train
```

启动器会自动进行 Lexmount/CDP 真实网站预检、准备数据、拉取 NeMo-RL v0.6 镜像（若缺失）并启动双卡容器。没有 Docker 权限时会要求输入 `sudo` 密码。

正式配置固定为：Qwen3-1.7B、150 optimizer steps、每 step 8 tasks × 8 rollouts = 64 条轨迹、global batch 64、micro batch 4（梯度累积 16）、12K context、每次 assistant 输出最多 512 tokens。vLLM 与 policy 均 TP=2，共用两张 GPU，并使用 NeMo-RL 原生 sleep/wake。

## 4. 查看结果

```bash
# 最新训练日志与轨迹审计
ls -lt logs/lexbrowser-grpo | head
tail -f logs/lexbrowser-grpo/train-*.attempt1.log

# TensorBoard
tensorboard --logdir logs/lexbrowser-grpo --host 0.0.0.0 --port 6006
```

正式 checkpoint 写入 `results/lexbrowser-grpo/train-<timestamp>/`；日志、trajectory audit 和 GPU 采样文件写入 `logs/lexbrowser-grpo/`。
