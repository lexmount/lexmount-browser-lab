# LexBrowser WebVoyager GRPO on H100 - from zero to the reward curve

This directory is a self-contained H100/CUDA port of the **validated**
Browser-RL training recipe: Qwen3-8B + verl GRPO + NeMo-Gym browser
environment sidecar + Lexmount cloud browser + WebVoyager tasks +
LLM-judge binary reward.

The recipe reproduced here is the exact configuration that produced the
reward-growth curve on 2x8 Ascend 910B on 2026-07-21 (reward mean
~0.105 over the first 10 steps -> ~0.289 over the last 10 steps, 60 steps
total). Every hyperparameter is kept identical; only the hardware/backend
layer changes (HCCL->NCCL, vllm-ascend->CUDA vLLM). See
[PORTING.md](PORTING.md) for the complete difference list.


### Reading the outputs

- Per-step rollout dumps land in `<run-dir>/rollouts/`; the reward field in those JSONL rows is named `score`. Judge inputs/outputs are audited in `<run-dir>/audit/judge_io.jsonl`.
- If every rollout in a step scores identically (all 0 or all 1), GRPO's group-relative advantage is zero and `grad_norm=0` for that step — this is expected zero-variance behavior, not a failure. The base model commonly produces such steps early on.
- Disk: each checkpoint is ~92 GB at world size 8 (`SAVE_FREQ=20` → 3 checkpoints over 60 steps); budget accordingly.

## Provenance

| Item | Value |
| --- | --- |
| Validated commit | `3220bc5f6f319c7421fcd1e196eb6d59fa190e8b` (internal; every file under `runtime/` is a byte-identical copy from it) |
| Validated runs | `20260721-120330-C-...-lexmount-deepseekv4-40k10t-60s-agentfix` (Lexmount backend) and `20260721-132000-D-...-local-deepseekv4-40k10t-60s` (local-Chromium control) |
| verl | 0.9.0.dev0, git commit `30119a253087bff86c12d329d2d8dd43c589705f` |
| vLLM | 0.18.0 (Ascend ran +vllm-ascend 0.18.1.dev41; CUDA-irrelevant) |
| torch | 2.9.0 on Ascend; 2.10.0+cu on CUDA (pinned by vLLM 0.18.0 wheels) |
| transformers | 5.3.0.dev0 on Ascend; 5.3.0 on CUDA |
| NeMo-Gym | v0.2.1, commit `27e921137042dcdb8a39c7169128619b9108074b` |
| Policy model | Qwen/Qwen3-8B (stock Hugging Face weights) |
| Judge model | `deepseek-v4-flash` via any OpenAI-compatible endpoint |
| Training data | 168-task `webvoyager-clean` set, `data/webvoyager-clean/MANIFEST.json` (in this subtree) |

## Architecture

```text
verl GRPO (8 or 16 H100)
  -> verl async vLLM rollout (TP=4, hermes multi-turn tools)
  -> lexbrowser_tool_agent agent loop (runtime/lexbrowser_verl_agent.py)
  -> HTTP -> NeMo-Gym sidecar (CPU-only, runtime/nemo_gym_webvoyager_server.py)
       -> WebVoyager environment (runtime/lexbrowser_webvoyager/)
       -> Lexmount cloud browser (CDP)
       -> OpenAI-compatible judge (deepseek-v4-flash)
  -> binary reward -> GRPO advantage -> FSDP update
```

Each step samples 8 tasks x 8 rollouts = 64 trajectories against real
websites, judges the final answers, and applies one GRPO update.

## Prerequisites

1. **Hardware**: one node with 8x H100-80GB (default), or two such nodes
   with password-less SSH from the head to the worker (`SSH_USER`/`SSH_KEY`
   overridable). ~200 GB free disk for checkpoints per run (a step-60 FSDP
   checkpoint set is ~100 GB at world size 16).
2. **Software**: Docker with the NVIDIA container toolkit; outbound
   network access to GitHub, PyPI/Docker Hub, Hugging Face, the Lexmount
   API and the public websites in the task set (arxiv.org, bbc.com,
   coursera.org, github.com).
3. **Credentials** (see `secrets.env.example` for details):
   - Lexmount API key + project ID (browser sessions; 64-session quota).
   - An OpenAI-compatible endpoint serving `deepseek-v4-flash` for the
     judge. Any provider or self-hosted deployment works; the endpoint we
     used internally is not reachable externally.
4. **Model**: stock Qwen3-8B weights from Hugging Face.

## Step-by-step

All commands run on the head node from the repository root.

### 1. Get the code and data

```bash
git clone https://github.com/lexmount/lexmount-browser-lab.git && cd lexmount-browser-lab
```

Everything this recipe needs is contained in the `training/h100/` subtree
(launch scripts, the vendored verl runtime under `training/h100/runtime/`,
and the task data under `training/h100/data/`).

The 168-task manifest `training/h100/data/webvoyager-clean/tasks.jsonl` is
tracked; the
training parquet is built automatically at first launch (or manually:
`python3 training/h100/build_webvoyager_clean_data.py`). Hashes for every
step of the data derivation are in
`training/h100/data/webvoyager-clean/MANIFEST.json`.

### 2. Build the training image

```bash
docker build -f training/h100/Dockerfile -t lexbrowser-verl-h100:local training/h100
```

This layers verl (pinned commit), transformers 5.3.0 and the environment
package on top of `vllm/vllm-openai:v0.18.0`. On a two-node setup, build
(or `docker save`/`load`) the same image on both nodes.

### 3. Download the policy model

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B
```

Any path works; pass it as `MODEL_PATH`. Two-node setups need the model at
the same path on both nodes.

### 4. Install the NeMo-Gym sidecar runtime (once per head node)

```bash
bash training/h100/install_nemo_gym_runtime_h100.sh
```

Clones NVIDIA-NeMo/Gym v0.2.1 (commit-verified) and installs its few extra
CPU dependencies into `WORK_ROOT` (default `/data/lexbrowser-rl`).

### 5. Configure secrets

```bash
cp training/h100/secrets.env.example training/h100/secrets.env
chmod 600 training/h100/secrets.env
$EDITOR training/h100/secrets.env   # Lexmount + judge credentials
```

### 6. Launch

Single node, 8x H100:

```bash
NODES_CSV=<this-host-ip> MODEL_PATH=/models/Qwen3-8B \
  bash training/h100/launch_h100.sh
```

Two nodes, 16x H100 (same world size as the validated 910B run):

```bash
NODES_CSV=<head-ip>,<worker-ip> MODEL_PATH=/models/Qwen3-8B \
  bash training/h100/launch_h100.sh
```

The launcher builds the parquet if missing, runs a per-node preflight
(GPUs, disk, image, CUDA, NeMo-Gym import, and a **real** Lexmount
session/CDP/DOM probe), starts the browser sidecar and Ray, waits for all
GPUs to register, and submits the 60-step run. Every hyperparameter above
can be overridden by env var, but the defaults are the validated
configuration - override nothing when the goal is curve reproduction.

Useful toggles: `SKIP_PREFLIGHT=1`, `STAMP=<run-id>`,
`RESUME_FROM_PATH=/workspace/checkpoints/<run-id>/global_step_<N>`,
`PPO_MAX_TOKEN_LEN_PER_GPU=12288` (exact validated packing; default 15360
is the 80 GB-scaled value, see PORTING.md).

### 7. Monitor

```bash
tail -f  <RUNS_ROOT>/<run-id>/logs/train.log
tensorboard --logdir <RUNS_ROOT>/<run-id>/tensorboard
docker logs -f lexbrowser-nemo-gym-webvoyager       # browser sidecar
```

The reward scalar is `critic/rewards/mean` (with per-step rollout JSONL
under `<run-dir>/rollouts/` and judge audit records under the sidecar
audit dir). After the run, `training/h100/runtime/verify_rollout_groups.py`
is invoked automatically to assert the 8x8 grouping was maintained.

## What the curve should look like

Reference numbers from the validated Lexmount-backend run (60 steps,
3840 rollouts):

| Metric | Value |
| --- | --- |
| Reward mean, steps 1-10 | 0.105 |
| Reward mean, steps 11-50 | 0.280 |
| Reward mean, steps 51-60 | 0.289 |
| Reward mean, all 60 steps | 0.252 |
| Positive-reward trajectories | 969 / 3840 |
| Zero-reward steps | 0 / 60 |

Expect the same shape: rewards near ~0.1 for the first handful of steps,
climbing to a ~0.25-0.30 plateau by roughly step 15-20, with high per-step
variance (single steps ranged from 1 to 40 positives out of 64). The
local-Chromium control run showed the same behavior (0.145 -> 0.272),
so the growth is a property of the recipe, not of one backend. The
original TensorBoard event archives for both validated runs (SHA256
`534390fc...` cloud, `8882e27a...` local) are archived with the 2026-07-21
experiment report and are available on request for side-by-side comparison.

**Wall-clock**: the validated run took ~4 h for 60 steps on 16x 910B
(238 s/step, of which ~87% was rollout/browser time, not GPU math). On
16x H100 expect the same order or faster (rollout time is dominated by
browser and judge latency, which do not change; generation and update
phases should be faster). On a single 8x H100 node, per-step rollout
throughput halves (32 concurrent trajectories per wave instead of 64),
so plan for roughly 6-8 h. The reward curve is step-indexed, so
single-node runs reproduce the same curve, just more slowly.

## Known differences vs. the 910B run (and why they don't matter)

- **HCCL -> NCCL, torch_npu -> CUDA torch**: collective transport and
  device runtime only; the verl patches are device-gated and identical.
- **vllm-ascend dropped**: it is an acceleration plugin for the same vLLM
  0.18 line; the CUDA build is the reference implementation.
- **torch 2.9.0 -> 2.10.0+cu**: forced by CUDA vLLM 0.18.0 wheels; no
  training-code dependency on 2.9-specific behavior.
- **Per-GPU token budget 12288 -> 15360 (default)**: dynamic-batching
  packing knob scaled for 80 GB; loss normalization makes it
  semantics-preserving, and `12288` remains available for exact parity.
- **1 node option**: identical per-step math at half the rollout
  concurrency.

Full details and the verification status of each claim: [PORTING.md](PORTING.md).

## File map

| File | Role |
| --- | --- |
| `launch_h100.sh` | one-command orchestrator (preflight -> sidecar -> Ray -> training) |
| `run_lexbrowser_grpo_h100.sh` | the verl trainer invocation (validated hyperparams as defaults) |
| `start_ray_node_h100.sh` | per-node Ray container (head/worker) |
| `start_nemo_gym_webvoyager_server_h100.sh` | CPU browser-environment sidecar |
| `install_nemo_gym_runtime_h100.sh` | one-time NeMo-Gym v0.2.1 runtime install |
| `preflight_h100.sh` | per-node checks incl. real Lexmount browser probe |
| `build_webvoyager_clean_data.py` | rebuilds the 168-task training set (hash-verified) |
| `Dockerfile` / `requirements-cuda.txt` | pinned CUDA software stack |
| `secrets.env.example` | every required credential, with sources |
| `PORTING.md` | Ascend->CUDA differences, verification status |
| `data/webvoyager-clean/` | 168-task manifest + hash chain (`MANIFEST.json`) |
| `runtime/` | vendored hardware-neutral verl runtime (agent loop, sidecar, patches, converters) from the validated internal commit |
