#!/usr/bin/env bash
# Launch verl GRPO from the Ray head inside the CUDA training container.
#
# CUDA port of the internal Ascend trainer entry. The training
# semantics (GRPO geometry, lengths, optimizer, multi-turn contract) are
# identical to the validated 2026-07-21 Ascend run; only the collective
# backend (HCCL -> NCCL) and per-GPU token budgets (64 GB NPU -> 80 GB H100)
# differ. See training/h100/PORTING.md.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$SCRIPT_DIR}
RUNS_ROOT=${RUNS_ROOT:-/workspace/runs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace/checkpoints}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
VERL_PROCESS_GROUP_TIMEOUT_SECONDS=${VERL_PROCESS_GROUP_TIMEOUT_SECONDS:-7200}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
DATA=${DATA:-$ROOT/data/webvoyager-clean/train.lexbrowser.parquet}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}

# --- Validated hyperparameters (2026-07-21, 60-step reward-growth run). ---
# Defaults below reproduce that run exactly; do not change them when the goal
# is curve reproduction.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
# TrainerV1 multiplies this prompt-level value by rollout.n internally.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
TOTAL_STEPS=${TOTAL_STEPS:-60}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-36864}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-40960}
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}
# Validated value was 12288 tokens per 64 GB Ascend 910B. H100-80GB has 1.25x
# the memory, so the default here is scaled to 15360 (= 12288 * 80/64). This
# is a throughput/packing knob for dynamic batching only - verl normalizes
# the loss across micro-batches, so it does not change training semantics.
# Set all three to 12288 to match the validated packing exactly.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-15360}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-15360}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-15360}
ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true}
ENTROPY_FROM_LOGITS_CHUNK_SIZE=${ENTROPY_FROM_LOGITS_CHUNK_SIZE:-256}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-10}
MAX_USER_TURNS=${MAX_USER_TURNS:-10}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-16384}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
EXPECTED_TASKS_PER_STEP=${EXPECTED_TASKS_PER_STEP:-8}
EXPECTED_ROLLOUTS_PER_TASK=${EXPECTED_ROLLOUTS_PER_TASK:-8}
RUN_DIR=$RUNS_ROOT/$STAMP
CHECKPOINT_DIR=$CHECKPOINT_ROOT/$STAMP

if [[ "$TRAIN_BATCH_SIZE" -ne "$EXPECTED_TASKS_PER_STEP" || "$ROLLOUT_N" -ne "$EXPECTED_ROLLOUTS_PER_TASK" ]]; then
  echo "WebVoyager GRPO geometry must be ${EXPECTED_TASKS_PER_STEP} tasks x ${EXPECTED_ROLLOUTS_PER_TASK} rollouts; got ${TRAIN_BATCH_SIZE} x ${ROLLOUT_N}." >&2
  exit 2
fi
if [[ "$PPO_MINI_BATCH_SIZE" -ne "$TRAIN_BATCH_SIZE" ]]; then
  echo "TrainerV1 PPO_MINI_BATCH_SIZE must equal the prompt batch (${TRAIN_BATCH_SIZE}); TrainerV1 multiplies it by rollout.n=${ROLLOUT_N}." >&2
  exit 2
fi
if ! [[ "$VERL_PROCESS_GROUP_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] ||
   (( VERL_PROCESS_GROUP_TIMEOUT_SECONDS < 1800 )); then
  echo "VERL_PROCESS_GROUP_TIMEOUT_SECONDS must be an integer >= 1800." >&2
  exit 2
fi
for value in \
  "$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  "$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  "$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  "$ENTROPY_FROM_LOGITS_CHUNK_SIZE"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Sequence-parallel, token-budget, and entropy chunk values must be positive integers." >&2
    exit 2
  fi
done
if (( NNODES * GPUS_PER_NODE % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "World size must be divisible by ULYSSES_SEQUENCE_PARALLEL_SIZE." >&2
  exit 2
fi
for budget in \
  "$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  "$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"; do
  if (( budget * ULYSSES_SEQUENCE_PARALLEL_SIZE < MAX_MODEL_LENGTH )); then
    echo "Each micro-batch token budget must fit one full-length trajectory after sequence parallelism." >&2
    exit 2
  fi
done
if [[ "$ENTROPY_FROM_LOGITS_WITH_CHUNKING" != "true" &&
      "$ENTROPY_FROM_LOGITS_WITH_CHUNKING" != "false" ]]; then
  echo "ENTROPY_FROM_LOGITS_WITH_CHUNKING must be true or false." >&2
  exit 2
fi

resume_args=()
if [[ -n "$RESUME_FROM_PATH" ]]; then
  expected_world_size=$((NNODES * GPUS_PER_NODE))
  if [[ ! -f "$RESUME_FROM_PATH/data.pt" ]]; then
    echo "Resume checkpoint is incomplete: missing $RESUME_FROM_PATH/data.pt" >&2
    exit 1
  fi
  for kind in model optim extra_state; do
    actual=$(find "$RESUME_FROM_PATH/actor" -maxdepth 1 -type f \
      -name "${kind}_world_size_${expected_world_size}_rank_*.pt" | wc -l)
    if [[ "$actual" -ne "$expected_world_size" ]]; then
      echo "Resume checkpoint is incomplete: $kind shards=$actual expected=$expected_world_size" >&2
      exit 1
    fi
  done
  resume_args=(
    trainer.resume_mode=resume_path
    trainer.resume_from_path="$RESUME_FROM_PATH"
  )
  echo "RESUME_CHECKPOINT_OK path=$RESUME_FROM_PATH world_size=$expected_world_size"
else
  echo "TRAINING_FROM_SCRATCH"
fi

# NCCL replaces HCCL from the Ascend recipe. GLOO remains the CPU-side
# coordination backend, exactly as on Ascend.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}
[[ -z "$NCCL_SOCKET_IFNAME" ]] && unset NCCL_SOCKET_IFNAME
[[ -z "$GLOO_SOCKET_IFNAME" ]] && unset GLOO_SOCKET_IFNAME
export VERL_PROCESS_GROUP_TIMEOUT_SECONDS
export PYTHONPATH="$ROOT/runtime:$ROOT/runtime/lexbrowser_webvoyager/src:${PYTHONPATH:-}"
# Keep TensorBoard event files on the mounted run directory rather than the
# container's relative working directory.
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}
export LEXBROWSER_METRICS_DIR=${LEXBROWSER_METRICS_DIR:-$RUN_DIR/observability/raw}
export VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-$RUN_DIR/metrics}
mkdir -p "$TENSORBOARD_DIR" "$LEXBROWSER_METRICS_DIR" "$VERL_FILE_LOGGER_ROOT" \
  "$RUN_DIR/rollouts" "$CHECKPOINT_DIR"
echo "WEBVOYAGER_LENGTHS model=$MAX_MODEL_LENGTH initial_prompt=4096 rollout=$MAX_RESPONSE_LENGTH action_per_turn=${LEXBROWSER_ACTION_MAX_TOKENS:-1024} tool_response_chars=$MAX_TOOL_RESPONSE_LENGTH reasoning_parser=$REASONING_PARSER"
echo "WEBVOYAGER_MEMORY sp=$ULYSSES_SEQUENCE_PARALLEL_SIZE ppo_tokens_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU ref_logprob_tokens_per_gpu=$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU old_logprob_tokens_per_gpu=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU entropy_chunking=$ENTROPY_FROM_LOGITS_WITH_CHUNKING entropy_chunk_size=$ENTROPY_FROM_LOGITS_CHUNK_SIZE"

python3 -m verl.trainer.main_ppo \
  hydra.run.dir="$RUN_DIR/hydra" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['$DATA']" \
  data.val_files="['$DATA']" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.tool_config_path="$ROOT/runtime/lexbrowser_tools.yaml" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="$ENTROPY_FROM_LOGITS_WITH_CHUNKING" \
  actor_rollout_ref.actor.entropy_from_logits_chunk_size="$ENTROPY_FROM_LOGITS_CHUNK_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser="$REASONING_PARSER" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT/runtime/lexbrowser_tools.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=lexbrowser_tool_agent \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.val_before_train=False \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.logger='["console","tensorboard","file"]' \
  trainer.project_name=lexbrowser_grpo_h100 \
  trainer.experiment_name="${MODEL_NAME}_${NNODES}n_${STAMP}" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  "${resume_args[@]}"

python3 "$ROOT/runtime/verify_rollout_groups.py" \
  --rollout-dir "$RUN_DIR/rollouts" \
  --tasks-per-step "$EXPECTED_TASKS_PER_STEP" \
  --rollouts-per-task "$EXPECTED_ROLLOUTS_PER_TASK"
