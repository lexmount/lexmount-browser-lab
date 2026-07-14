#!/usr/bin/env bash
# The only process that starts the NeMo GRPO driver after the Ray cluster is ready.
set -Eeuo pipefail

RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in --run-dir) RUN_DIR="$2"; shift 2 ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac
done
[[ -n "$RUN_DIR" ]] || { echo "--run-dir is required" >&2; exit 2; }

ROOT="/workspace/LexBrowserEnv"
DELIVERY_DIR="$ROOT/training/nvidia"
GYM_DIR="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/verifiers_agent"
CONFIG="$DELIVERY_DIR/configs/grpo_lexbrowser_webvoyager_qwen3_1_7b_nvidia.yaml"
SOURCE_DATA="$ROOT/training/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/datasets/WebVoyager_data_clean.jsonl"
COMPLETED=0

# shellcheck disable=SC1090
source "$RUN_DIR/run.env"
# Credentials remain in the private runner file and are intentionally never
# copied into run.env, the model cache, reports, or the run manifest.
set +x
# shellcheck disable=SC1091
source "$ROOT/secrets.env"

phase() { python3 "$DELIVERY_DIR/scripts/run_manifest.py" phase --run-dir "$RUN_DIR" --name "$1" --status "$2" --detail "$3"; }
cleanup() {
  status=$?
  set +e
  python3 "$DELIVERY_DIR/scripts/node_metrics.py" stop --run-dir "$RUN_DIR"
  python3 "$DELIVERY_DIR/scripts/summarize_run.py" --run-dir "$RUN_DIR"
  mkdir -p "$RUN_DIR/diagnostics"
  cp "$RUN_DIR/preflight/summary.json" "$RUN_DIR/diagnostics/preflight_summary.json" 2>/dev/null || true
  cp "$RUN_DIR/metrics/resources_summary.json" "$RUN_DIR/diagnostics/resources_summary.json" 2>/dev/null || true
  if [[ "$status" -eq 0 && "$COMPLETED" -eq 1 ]]; then
    python3 "$DELIVERY_DIR/scripts/run_manifest.py" finalize --run-dir "$RUN_DIR" --status complete --detail "training driver completed"
  else
    python3 "$DELIVERY_DIR/scripts/run_manifest.py" finalize --run-dir "$RUN_DIR" --status failed --detail "training driver exited with status $status"
  fi
  exit "$status"
}
trap cleanup EXIT

for key in LEXMOUNT_BASE_URL LEXMOUNT_API_KEY LEXMOUNT_PROJECT_ID OPENAI_API_KEY OPENAI_BASE_URL; do
  [[ -n "${!key:-}" ]] || { phase credentials failed "required secret name is empty: $key"; exit 1; }
done

phase dataset started "validating and preprocessing the revision-pinned WebVoyager source"
mkdir -p "$RUN_DIR/data"
python3 "$ROOT/training/scripts/prepare_webvoyager_data.py" --source "$SOURCE_DATA" --output "$RUN_DIR/data/train.jsonl" --manifest "$RUN_DIR/data/train_manifest.json"
python3 "$ROOT/training/scripts/prepare_webvoyager_data.py" --source "$SOURCE_DATA" --output "$RUN_DIR/data/smoke.jsonl" --manifest "$RUN_DIR/data/smoke_manifest.json" --limit 1 --offset "${WEBVOYAGER_SMOKE_TASK_OFFSET:-0}"
phase dataset complete "source SHA, row count, task uniqueness, and prepared JSONL recorded"

phase model started "downloading the requested Hugging Face revision into the shared run artifact"
python3 "$DELIVERY_DIR/scripts/download_model.py" --model-id "$LEXBROWSER_MODEL_ID" --revision "$LEXBROWSER_MODEL_REVISION" --output "$RUN_DIR/model" --manifest "$RUN_DIR/manifests/model.json"
model_revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_revision"])' "$RUN_DIR/manifests/model.json")"
model_hash="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_sha256"])' "$RUN_DIR/manifests/model.json")"
python3 "$DELIVERY_DIR/scripts/run_manifest.py" model --run-dir "$RUN_DIR" --revision "$model_revision" --snapshot-sha256 "$model_hash" --path "$RUN_DIR/model"
phase model complete "model revision and snapshot hash recorded"

phase metrics started "starting one detached CPU/RAM/GPU/disk/network sampler per Ray node"
python3 "$DELIVERY_DIR/scripts/node_metrics.py" start --run-dir "$RUN_DIR" --expected-nodes "$LEXBROWSER_NODES" --interval "${LEXBROWSER_METRICS_INTERVAL_SECONDS:-5}"
phase metrics complete "node samplers started"

phase browser_preflight started "checking a real Lexmount Chrome/CDP navigation before scheduling rollouts"
preflight_url="${LEXBROWSER_PREFLIGHT_URL:-https://www.apple.com/}"
"$GYM_DIR/.venv/bin/python" "$ROOT/training/scripts/smoke_lexmount_cdp.py" --url "$preflight_url" --timeout 60 2>&1 | tee "$RUN_DIR/logs/lexmount_cdp_preflight.log"
phase browser_preflight complete "real-site Lexmount CDP preflight passed"

data_path="$RUN_DIR/data/train.jsonl"
checkpoint_dir="$RUN_DIR/checkpoints"
overrides=(
  "policy.model_name=$RUN_DIR/model"
  "policy.tokenizer.name=$RUN_DIR/model"
  "policy.dtensor_cfg.tensor_parallel_size=$LEXBROWSER_GPUS_PER_NODE"
  "policy.generation.vllm_cfg.tensor_parallel_size=$LEXBROWSER_GPUS_PER_NODE"
  "policy.generation.colocated.resources.gpus_per_node=$LEXBROWSER_GPUS_PER_NODE"
  "policy.generation.colocated.resources.num_nodes=$LEXBROWSER_NODES"
  "cluster.gpus_per_node=$LEXBROWSER_GPUS_PER_NODE"
  "cluster.num_nodes=$LEXBROWSER_NODES"
  "data.train.data_path=$data_path"
  "checkpointing.checkpoint_dir=$checkpoint_dir"
  "logger.log_dir=$RUN_DIR/logs/tensorboard"
)
if [[ -n "${LEXBROWSER_RESUME_FROM:-}" ]]; then
  checkpoint_dir="$LEXBROWSER_RESUME_FROM"
  overrides+=("checkpointing.checkpoint_dir=$checkpoint_dir")
  phase resume started "explicit checkpoint directory: $checkpoint_dir"
fi
if [[ "$LEXBROWSER_MODE" == "smoke" ]]; then
  data_path="$RUN_DIR/data/smoke.jsonl"
  overrides+=(
    "grpo.num_prompts_per_step=1" "grpo.num_generations_per_prompt=2"
    "grpo.max_num_steps=1" "grpo.max_num_epochs=1"
    "policy.train_global_batch_size=2" "policy.train_micro_batch_size=1"
    "policy.generation_batch_size=2" "data.train.data_path=$data_path"
    "checkpointing.enabled=false"
  )
fi

export LEXBROWSER_TRAJECTORY_AUDIT_LOG="$RUN_DIR/metrics/trajectory_audit.jsonl"
export PYTHONPATH="$ROOT/training/nemo_gym/runtime_overrides${PYTHONPATH:+:$PYTHONPATH}"
phase training started "launching NeMo RL GRPO with ${LEXBROWSER_NODES}x${LEXBROWSER_GPUS_PER_NODE} topology"
attempt=1
while true; do
  attempt_log="$RUN_DIR/logs/training-attempt${attempt}.log"
  set +e
  (cd /opt/nemo-rl && python examples/nemo_gym/run_grpo_nemo_gym.py --config "$CONFIG" "${overrides[@]}") 2>&1 | tee "$attempt_log"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -eq 0 ]]; then
    break
  fi
  if [[ "$attempt" -eq 1 ]] && grep -q "Error in memory profiling" "$attempt_log"; then
    phase training started "retrying the documented vLLM memory-profiling race once"
    attempt=2
    sleep 30
    continue
  fi
  phase training failed "attempt $attempt failed; non-transient failures are not silently retried"
  exit "$status"
done
phase training complete "NeMo RL exited successfully"

if [[ "$LEXBROWSER_MODE" == "train" ]]; then
  phase report started "extracting TensorBoard curves and browser trajectory timing"
  python "$ROOT/training/scripts/generate_train_report.py" --log-root "$RUN_DIR/logs/tensorboard" --output-dir "$RUN_DIR/reports/training" --training-log "$RUN_DIR/logs/training-attempt${attempt}.log" --audit-log "$RUN_DIR/metrics/trajectory_audit.jsonl" --group-size 64 --mode train
  phase report complete "reward curve and trajectory report generated"
fi
COMPLETED=1
