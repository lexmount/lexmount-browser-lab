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

## 5. 训练后 checkpoint 对比

`training/scripts/webvoyager_posttrain_eval.py` 复刻 910B Qwen3-8B WebVoyager
训练时的策略接口：单个 `browser(operation, instruction)` 工具、CDP DOM、每回合
最多 1024 tokens、最多 6 个 assistant 回合。两端只替换浏览器会话实现，因此可用于
训练后模型的 Lexmount / 5090 Local Chrome 成对比较。

先从训练 parquet 与完整 WebVoyager 任务集生成固定清单。输出包含 20-task smoke、
100-task training-overlap 和 43-task holdout；600 个 overlap 只用于回放/环境等价性，
43 个 holdout 都来自 Cambridge Dictionary，不能视为广泛泛化结论。

```bash
PYTHONPATH=training/lexbrowser_webvoyager/src \
  /home/wf/sxh/lexmount-browser-lab-eval/.venv-webvoyager/bin/python \
  training/scripts/webvoyager_posttrain_eval.py prepare-splits \
  --training-parquet /data/wf/sxh/workspace/nemorl-webagent/training/data/webvoyager/verl-train-task-only.parquet \
  --benchmark-jsonl /home/wf/sxh/browseruse-agent-bench/browseruse_bench/data/WebVoyager/task.jsonl \
  --output-dir /data/wf/sxh/webvoyager-posttrain/splits
```

待 checkpoint 的 `model.safetensors` 校验完成后，先各跑同一 smoke 清单。`eval.env`
仅在运行机保存，包含 policy/Judge/Lexmount 的既有变量；原始轨迹和资源采样也只留在
运行机。`--judge training` 使用训练期的 evidence/final-answer judge；smoke 可先用
`--judge off` 确认工具协议和浏览器路径。

```bash
EVAL_PY=/home/wf/sxh/lexmount-browser-lab-eval/.venv-webvoyager/bin/python
COMMON=(
  --tasks /data/wf/sxh/webvoyager-posttrain/splits/smoke_20.jsonl
  --model qwen3-8b-webvoyager-grpo-step150
  --model-artifact /data/wf/sxh/webrl_trained_models/qwen3-8b-webvoyager-grpo-global_step_150-hf
  --model-sha256 e986f56675a265e96bc7ad52992e0c317967bebec26a599f5a9dba9f8a3355b7
  --policy-base-url http://127.0.0.1:18088/v1
  --env-file /data/wf/sxh/webvoyager-posttrain/eval.env
  --judge training
)

PYTHONPATH=training/lexbrowser_webvoyager/src "$EVAL_PY" \
  training/scripts/webvoyager_posttrain_eval.py run \
  "${COMMON[@]}" --backend local --local-chrome-executable /usr/bin/google-chrome \
  --output-dir /data/wf/sxh/webvoyager-posttrain/runs/step150-local-smoke

PYTHONPATH=training/lexbrowser_webvoyager/src "$EVAL_PY" \
  training/scripts/webvoyager_posttrain_eval.py run \
  "${COMMON[@]}" --backend lexmount \
  --output-dir /data/wf/sxh/webvoyager-posttrain/runs/step150-lexmount-smoke
```

`summary.json` 中的 `statuses.completed` 只表示 runner 已结束；模型效果以
`judge.success_rate` 为准，并结合 `final_answer_statuses`、`trajectory` 与原始 JSONL
区分策略失败和基础设施失败。不要把 `completed` 当作成功率。

在解释模型效果前，先跑同一清单的 browser availability probe。它不调用 policy、不执行
用户任务动作，只验证 `fresh session -> start_url -> observe` 是否获得可用 DOM，因此能把
浏览器/出口稳定性与 checkpoint 能力分开。Lexmount 的代理模式是实验变量，必须在 manifest
中保留；例如下列命令明确使用官方代理。

```bash
PYTHONPATH=training/lexbrowser_webvoyager/src "$EVAL_PY" \
  training/scripts/webvoyager_posttrain_eval.py probe \
  --tasks /data/wf/sxh/webvoyager-posttrain/splits/smoke_20.jsonl \
  --backend local --local-chrome-executable /usr/bin/google-chrome \
  --per-tool-timeout 25 \
  --output-dir /data/wf/sxh/webvoyager-posttrain/probes/local-smoke

PYTHONPATH=training/lexbrowser_webvoyager/src "$EVAL_PY" \
  training/scripts/webvoyager_posttrain_eval.py probe \
  --tasks /data/wf/sxh/webvoyager-posttrain/splits/smoke_20.jsonl \
  --backend lexmount --env-file /data/wf/sxh/webvoyager-posttrain/eval.env \
  --lexmount-official-proxy --per-tool-timeout 25 \
  --output-dir /data/wf/sxh/webvoyager-posttrain/probes/lexmount-official-smoke
```

用已有 `src/lexbrowser_eval/resources/cgroup_profiler.py` 包裹上述命令，可同时保存
CPU、PSS、Chrome PSS、GPU、显存和 vLLM 队列采样，资源指标口径与 LexBench 压力实验一致。
5090 上应传入 `--gpu-index 0`，避免另一张 GPU 的负载污染统计；该 profiler 会在 tmux/SSH
后台运行时自动恢复用户级 systemd bus。
