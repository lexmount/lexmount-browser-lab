# Ascend 910B -> H100 Porting Notes

This document records every deliberate difference between the validated
Ascend 910B run (internal commit `3220bc5`, 2026-07-21) and the H100/CUDA
port in `training/h100/`, plus what has and has not been verified.

The port was validated and packaged in the internal Ascend training
repository; this public subtree is a self-contained transplant of it. Every
runtime file it needs is vendored byte-for-byte from the internal validated
commit under `runtime/` (list below), so nothing outside `training/h100/`
is required.

## What the port changes

| Area | Validated 910B run | H100 port | Rationale |
| --- | --- | --- | --- |
| Collectives | HCCL (`HCCL_*` env, `vllm_ascend` PyHcclCommunicator) | NCCL (`NCCL_SOCKET_IFNAME`; verl's device abstraction picks NCCL automatically) | The verl patches in `runtime/patches/` are already device-agnostic: they gate on `is_npu_available` and fall back to CUDA/NCCL. No code change needed, only env plumbing. |
| Device plumbing | `/dev/davinci*` mounts, `ASCEND_RT_VISIBLE_DEVICES`, `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES`, `VLLM_ASCEND_ENABLE_NZ` | `docker --gpus all`; Ascend-only env vars dropped | Standard NVIDIA container runtime. |
| Base image | `quay.io/ascend/vllm-ascend:nightly-releases-v0.18.0` (vLLM 0.18.0 + vllm-ascend 0.18.1.dev41) | `vllm/vllm-openai:v0.18.0` | vllm-ascend 0.18.1 tracks vLLM 0.18.0; on CUDA the plugin drops out and the official vLLM 0.18.0 image is the direct equivalent. |
| torch | 2.9.0 (+ torch_npu; Ascend pin) | 2.10.0+cu (pinned by the vLLM 0.18.0 CUDA wheels) | On Ascend, torch was held at 2.9.0 by torch_npu. CUDA vLLM 0.18.0 requires torch==2.10.0; there is no CUDA vLLM 0.18.0 build against torch 2.9.0. Minor-version torch difference, same model/optimizer code paths. |
| transformers | 5.3.0.dev0 | 5.3.0 (PyPI release) | Closest released version to the validated dev snapshot. Note: PyPI vLLM 0.18.0 declares `transformers<5`; the validated stack already ran transformers 5.x over vLLM 0.18 (the Ascend nightly image shipped it), so the port follows the validated stack. Must be confirmed by the first smoke run. |
| verl | 0.9.0.dev0 built from git commit `30119a253087bff86c12d329d2d8dd43c589705f` | identical commit | 0.9.0.dev0 is not on PyPI; the git commit is the authoritative pin (it is the same `VERL_COMMIT` the Ascend Dockerfile used). |
| Per-GPU dynamic token budgets | 12288 (64 GB HBM) | 15360 default (= 12288 x 80/64), override to 12288 for exact-packing parity | `ppo_max_token_len_per_gpu` and the two log-prob budgets only control how verl packs sequences into micro-batches under `use_dynamic_bsz`; verl normalizes the loss across micro-batches, so this is a throughput knob, not a semantics change. The budget-x-SP >= max_model_len invariant holds for both values (49152 / 61440 >= 40960). |
| Topology | 2 nodes x 8 NPU | 1 node x 8 H100 (default) or 2 x 8 | GRPO geometry (8 tasks x 8 rollouts) and Ulysses SP=4 / vLLM TP=4 are unchanged and divide both world sizes. On 8 GPUs verl simply runs half as many concurrent vLLM engines; per-step math is identical, wall-clock per step is roughly doubled. |
| Node paths | site-specific `/data*/wf/...` defaults | neutral `WORK_ROOT=/data/lexbrowser-rl` defaults, all overridable | No behavioral change. |

## What is intentionally identical

- GRPO geometry: 8 tasks/step x 8 rollouts/task, ppo_mini_batch_size=8.
- 60 steps, 4 epochs, checkpoint every 20 steps, lr 5e-6 constant, KL loss
  coef 0.001 (low_var_kl), no KL in reward.
- Lengths: 40960 context, 4096 initial prompt, 36864 trajectory, 10+10
  turns, 1024 tokens/action, 16384-char tool responses, qwen3 reasoning
  parser.
- Entropy from logits with chunking, chunk size 256.
- FSDP full-parameter with param/optimizer offload, Ulysses SP=4,
  gradient checkpointing, remove-padding.
- vLLM: async rollout mode, TP=4, gpu_memory_utilization 0.30,
  hermes multi-turn tool format, `lexbrowser_tool_agent` agent loop.
- Environment: NeMo-Gym v0.2.1 (commit `27e92113...`) sidecar +
  `lexbrowser_webvoyager_no_anti_bot` env + Lexmount cloud browser,
  64 concurrent sessions / 16 concurrent creates.
- Judge: deepseek-v4-flash, temperature 0, binary yes/no reward.
- Data: the same 168-task webvoyager-clean set (see
  `data/webvoyager-clean/MANIFEST.json` for the full hash chain).
- The two verl patch files (`runtime/patches/agent_loop.py`,
  `runtime/patches/distributed.py`) are bind-mounted over the verl
  install unchanged - they are hardware-neutral.

## Vendored (hardware-neutral) runtime under training/h100/runtime/

The following files are part of the validated recipe and contain no Ascend
dependency. They are vendored byte-for-byte from the internal validated
commit `3220bc5` so the subtree is self-contained; the launch scripts
reference them under `runtime/`:

- `runtime/lexbrowser_verl_agent.py` - verl multi-turn agent loop
  (registers `lexbrowser_tool_agent`).
- `runtime/lexbrowser_tools.yaml` - tool schema for the agent loop.
- `runtime/nemo_gym_webvoyager_server.py` - CPU sidecar HTTP server.
- `runtime/rollout_audit.py` - judge-audit JSONL writer imported by the
  sidecar server.
- `runtime/patches/agent_loop.py`, `runtime/patches/distributed.py`
  - verl overlays (device-gated, CUDA-safe).
- `runtime/prepare_webvoyager_verl_data.py` - jsonl -> parquet converter.
- `runtime/verify_rollout_groups.py` - post-run rollout audit.
- `runtime/smoke_lexmount_cdp.py` - real-browser preflight probe.
- `runtime/lexbrowser_webvoyager/` - the environment package itself
  (`pyproject.toml`, `lexbrowser_webvoyager_no_anti_bot` sources, and the
  600-task `WebVoyager_data_clean.jsonl` dataset it ships).

This is the exact minimal closure of what the launcher chain executes,
sources, or imports. Internal-only helpers that the chain never touches
(observability exporters, an optional standalone sidecar smoke script,
and an unused reference-answer JSON that no vendored code path reads)
are intentionally not vendored.

Transplant-only mechanical changes (no semantic effect): the scripts'
`ROOT` now resolves to the `training/h100/` subtree instead of a repository
root, the container mount point is `/workspace/lexbrowser-h100`, vendored
paths moved from `training/ascend/...` and `training/scripts/...` to
`runtime/...`, and the task data lives at `training/h100/data/`. Every
hyperparameter, patch byte, and runtime file is otherwise identical.

## Statically verified on macOS (no CUDA available)

- `bash -n` on every new shell script.
- `python3 -m py_compile` on every new/reused Python entry point.
- The 168-task data chain verified end to end: tracked 600-task jsonl
  (sha `b901adc3...`) -> 4-site filter -> byte-identical `tasks.jsonl`
  (sha `db0dd8c1...`) -> converter runs and emits 168 task-only rows.
- verl commit `30119a25...` exists upstream and its version file reads
  `0.9.0.dev`; `vllm==0.18.0` exists on PyPI and Docker Hub
  (`vllm/vllm-openai:v0.18.0`); `transformers==5.3.0` exists on PyPI.

## Not verified until an H100 machine is available

1. Docker image build (network pulls, dependency resolution inside the
   image; in particular transformers 5.3.0 over CUDA vLLM 0.18.0).
2. vLLM 0.18.0 CUDA async-rollout + verl 0.9.0.dev0 sleep/wake and weight
   sync on H100 (validated only via vllm-ascend on 910B).
3. The 15360 token budget headroom on 80 GB (fallback: 12288, the
   validated value, which is guaranteed to fit in 64 GB and therefore
   also in 80 GB at equal or better margins).
4. NCCL multi-node transport settings for the 2-node variant (the
   `NCCL_IB_*` pass-throughs are provided but fabric-specific).
5. End-to-end reward-curve reproduction (the point of the exercise).
6. Byte-identity of the rebuilt parquet with the recorded
   `cc5d673d...` SHA (depends on the pyarrow version inside the image;
   row content is deterministic either way).

## Known non-portable pieces (documented, not guessed)

- The internal Ascend harness's launcher/node scripts, its observability
  stack, and its `npu-smi`-based MFU/power exporters are 910B-specific and
  are NOT part of this port (they stay in the internal repository).
  TensorBoard logging (reward curve) is built into verl and works
  unchanged on CUDA.
- `training/nemo_gym/`, `training/nemo_rl_patches/`, `training/nvidia/`
  and the rest of this repository's `training/` tree belong to a separate
  NeMo-RL-based harness, not to this validated verl recipe. They are
  unrelated to `training/h100/`, and this subtree does not reference them
  (it vendors its own copy of the environment package, pinned to the
  validated commit).
