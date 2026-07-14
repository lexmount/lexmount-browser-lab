# NVIDIA delivery package

This directory turns the validated WebVoyager/Lexmount GRPO path into a
Slurm-deliverable run. It leaves the existing 2x5090 launcher unchanged.
The default run is Qwen3-1.7B, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with the same 64-trajectory GRPO
update geometry that produced the stable reward signal: 8 prompts x 8 sampled
browser trajectories, seed 42, 12,288 context tokens, and 8 browser turns.

## What the runner does

1. Creates a secret-free immutable run manifest with git/config/dataset/model
   identifiers and the comparison contract.
2. Allocates Slurm nodes, verifies the requested NVIDIA GPU family, GPU count,
   free memory, and shared artifact storage on every node.
3. Builds the Lexmount/NeMo Gym verifier environment once on that shared
   storage, then imports it from every node.
4. Runs a full-topology NCCL all-reduce before starting Ray or browser work.
5. Verifies the pinned WebVoyager source (SHA, 600 rows, unique task IDs),
   prepares train/smoke JSONL, and downloads the pinned model snapshot.
6. Starts the official NeMo RL v0.6 Ray/Slurm launcher, gates rollouts on a
   real Lexmount CDP navigation, trains with checkpointing, and retries only
   the documented vLLM memory-profiling race once.
7. Emits TensorBoard/reward/trajectory reports and CPU, RAM, GPU, disk, and
   network samples from every Ray node.

There is no silent fallback to fewer GPUs, local Chrome, or a different model.
Any missing GPU, NCCL, shared-storage, browser, or training condition fails
the run and leaves a diagnosis in the run directory.

The delivery launcher mounts the current verifier config and editable
WebVoyager environment rather than maintaining a second browser lifecycle
implementation. It therefore inherits the validated 60-second Lexmount
session-create deadline and the late-session close/delete cleanup: timed-out
creates do not silently leave provider browser capacity behind.

## Cluster contract

- Slurm allocation supporting `srun --container-image` and
  `--container-mounts` (Pyxis/Enroot).
- Default formal topology: 8 homogeneous nodes x 8 GPUs/node. Override
  `--nodes` and `--gpus-per-node` for the target cluster; the driver updates
  tensor parallelism and NeMo resource topology from these values.
- `--gpu-family` defaults to `H100`, but can be set to `H200`, `A100`, or
  another NVIDIA family. Use `any` only when a family match cannot be asserted.
  `LEXBROWSER_MIN_FREE_MIB` controls the corresponding free-memory gate.
- A shared read/write filesystem visible at the repository and run-root paths.
- Inter-node networking/RDMA configured so the NCCL all-reduce can pass.
- Outbound Internet access over DNS and HTTPS is mandatory; an internal-only
  cluster is insufficient. The runner must reach NGC, PyPI, Hugging Face,
  Lexmount, the judge endpoint, and the real WebVoyager sites used by the run.
- Sufficient shared disk for the model snapshot, checkpoints, Ray logs, and
  browser trajectory audit. `secrets.env` must be visible beside the checkout,
  have mode `0600`, and must never be committed.

The packaged WebVoyager source data is already versioned in this repository;
the preprocessing step validates its canonical SHA-256 before use. The model
is downloaded at its pinned Hugging Face revision into the run artifact and
its resolved revision plus tree hash are written to `manifests/model.json`.

## Private credentials

```bash
cd LexBrowserEnv
cp training/nvidia/secrets.env.example secrets.env
chmod 600 secrets.env
```

Fill in the Lexmount and completion-judge values locally. The scheduler command
uses an explicit non-secret environment whitelist and does not carry these
values. `run.env`, manifests, preflight reports, resource reports, Ray logs,
and diagnostic bundles contain names/statuses only, never credential values.
The vendored NeMo launcher removes its upstream bare `env` diagnostics for the
same reason.

## Run sequence

The submission host needs `sbatch`; no interactive SSH into worker nodes is
required. If a managed platform invokes the script from an existing Slurm
allocation, the launcher detects `SLURM_JOB_ID` and uses that allocation rather
than nesting `sbatch`.

```bash
# Validate packaging only: creates a manifest, makes no network or Slurm call.
./training/nvidia/run_nvidia.sh --mode dry-run --run-root /shared/lexbrowser-runs

# One-node functional gate: GPU node check, Gym imports, 8-rank NCCL, one real
# browser preflight, one 1-prompt x 2-rollout GRPO update.
./training/nvidia/run_nvidia.sh --mode smoke --nodes 1 --gpus-per-node 8 --gpu-family H100 --run-root /shared/lexbrowser-runs

# Eight-node hardware/network gate only: no credentials, model, or training.
./training/nvidia/run_nvidia.sh --mode node-check --nodes 8 --gpus-per-node 8 --gpu-family H100 --run-root /shared/lexbrowser-runs

# Formal NVIDIA training. Defaults are 8 nodes x 8 GPUs. --wait is default;
# add --no-wait to only submit.
./training/nvidia/run_nvidia.sh --mode train --nodes 8 --gpus-per-node 8 --gpu-family H100 --run-root /shared/lexbrowser-runs
```

Optional scheduler configuration is supplied as non-secret environment values:
`LEXBROWSER_SLURM_PARTITION`, `LEXBROWSER_SLURM_ACCOUNT`,
`LEXBROWSER_SLURM_TIME`, `LEXBROWSER_SLURM_GRES`, and `NEMO_RL_IMAGE`.
To resume, pass an explicit checkpoint directory only:

```bash
./training/nvidia/run_nvidia.sh --mode train --resume /shared/lexbrowser-runs/train-.../checkpoints
```

## Evidence produced

Each run writes one directory under `--run-root`:

- `manifests/run_manifest.json`: git revision, config/dataset hashes, topology,
  secret-name presence, phases, and pinned model facts.
- `preflight/`: per-node GPU-family/free-memory/shared-output facts and the NCCL
  all-reduce result.
- `data/`: validated preprocessing manifests.
- `logs/`: Slurm, Ray, CDP preflight, TensorBoard, and per-attempt training logs.
- `metrics/nodes/*.jsonl`: raw 5-second CPU/RAM/GPU/disk/network samples.
- `metrics/resources_summary.json`: aggregated observed node-resource metrics
  plus parsed reward/loss signals.
- `metrics/trajectory_audit.jsonl` and `reports/training/`: browser timing,
  reward curves, and the existing report generator output for formal training.
- `diagnostics/`: a small safe-to-share subset of preflight/resource summaries.

The current training report contains online reward points from real GRPO
rollouts. It does not mislabel those as a held-out benchmark. A true backend
comparison must use independently collected runs under the exact contract
below.

## Lexmount vs local Chrome comparison

`scripts/compare_backends.py` compares two completed artifact directories:

```bash
python3 training/nvidia/scripts/compare_backends.py \
  --lexmount-run /shared/lexbrowser-runs/lexmount-train \
  --local-run /shared/lexbrowser-runs/local-chrome-train \
  --output /shared/lexbrowser-runs/backend-comparison.json
```

It refuses to report a comparison unless both manifests agree on the
`comparison_contract`, topology, model ID and resolved revision, and dataset
SHA. The local-Chrome trainer is deliberately **not** fabricated in this
package: it needs to produce a separately labelled `backend: local` manifest
under the same contract. Until that run exists, the only correct conclusion is
that a backend comparison is not yet comparable.

## Local validation

```bash
python3 -m unittest tests.training.test_nvidia_delivery
for script in training/nvidia/*.sh training/nvidia/scripts/*.sh; do bash -n "$script"; done
```
