#!/usr/bin/env bash
set -euo pipefail

# Optional: enable bash trace with DEBUG=1
if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

############################
# 1) User configuration
############################

# Repo root (auto-detected from this script location)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Logging / W&B (keep keys out of the script)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
else
  unset WANDB_API_KEY
fi
if [[ -n "${WANDB_BASE_URL:-}" ]]; then
  export WANDB_BASE_URL
else
  unset WANDB_BASE_URL
fi
export WANDB_MODE="${WANDB_MODE:-offline}"  # offline | online

# Runtime / NCCL
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-True}"

# Reward / safety settings
export REWARD_MODEL_USE_ANSWER_TAGS="${REWARD_MODEL_USE_ANSWER_TAGS:-0}"
export REWARD_HARM="${REWARD_HARM:-1}"
export REWARD_REFUSAL="${REWARD_REFUSAL:-0.5}"
export REWARD_LABEL="${REWARD_LABEL:-0}"
export USE_Q_SAFE_LABEL_FOR_REFUSAL="${USE_Q_SAFE_LABEL_FOR_REFUSAL:-1}"

# Safety score / safety endpoints (set these in your environment if needed)
export SAFETY_SCORE_MODE="${SAFETY_SCORE_MODE:-classifier}"  # classifier | rule_api
export WILDGUARD_API_ENDPOINT="${WILDGUARD_API_ENDPOINT:-}"
export SAFETY_RULE_API_ENDPOINT="${SAFETY_RULE_API_ENDPOINT:-}"
export SAFETY_RULE_API_KEY="${SAFETY_RULE_API_KEY:-}"
export SAFETY_RULE_MODEL_NAME="${SAFETY_RULE_MODEL_NAME:-}"
export DEFENDER_QUALITY_API_ENDPOINT="${DEFENDER_QUALITY_API_ENDPOINT:-}"
export REVISION_SCORE_ENDPOINT="${REVISION_SCORE_ENDPOINT:-${REVISION_ENDPOINT:-}}"

# Prompt templates (relative to repo)
export SAFETY_RULE_SYSTEM_PROMPT_PATH="${SAFETY_RULE_SYSTEM_PROMPT_PATH:-$PROJECT_DIR/src/verl/verl/utils/reward_score/defender_judge_system_prompt.txt}"
export SAFETY_RULE_PROMPT_PATH="${SAFETY_RULE_PROMPT_PATH:-$PROJECT_DIR/src/verl/verl/utils/reward_score/defender_judge_user_prompt.txt}"
export SAFETY_RULE_SAFE_SYSTEM_PROMPT_PATH="${SAFETY_RULE_SAFE_SYSTEM_PROMPT_PATH:-$PROJECT_DIR/src/verl/verl/utils/reward_score/defender_helpful_system_prompt.txt}"
export SAFETY_RULE_SAFE_PROMPT_PATH="${SAFETY_RULE_SAFE_PROMPT_PATH:-$PROJECT_DIR/src/verl/verl/utils/reward_score/defender_judge_user_prompt.txt}"
export SAFETY_RULE_SCORE_MIN="${SAFETY_RULE_SCORE_MIN:-0}"
export SAFETY_RULE_SCORE_MAX="${SAFETY_RULE_SCORE_MAX:-10}"
export REWARD_SCORE_MAX_WORKERS="${REWARD_SCORE_MAX_WORKERS:-1}"
export REWARD_SCORE_TIMEOUT_S="${REWARD_SCORE_TIMEOUT_S:-30}"

# Revision reward (optional)
export REVISION_MODEL_PATH="${REVISION_MODEL_PATH:-}"
export REVISION_MODEL_DEVICE="${REVISION_MODEL_DEVICE:-cpu}"
export ENABLE_REVISION_REWARD="${ENABLE_REVISION_REWARD:-0}"
export ENABLE_DEFENDER_QUALITY_REWARD="${ENABLE_DEFENDER_QUALITY_REWARD:-0}"
export FORMAT_REWARD_VALUE="${FORMAT_REWARD_VALUE:-1}"

# Ray
export RAY_MASTER_PORT="${RAY_MASTER_PORT:-6379}"

# Dataset paths (relative to repo)
TRAIN_FILES="${TRAIN_FILES:-data/safety/train.parquet}"
VAL_FILES="${VAL_FILES:-data/safety/test_wjb.parquet}"

# Output / checkpoints
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}"

# Model paths (set your own; no absolute paths here)
# Defender base + attacker SFT
MODEL_DEFENDER_BASE="${MODEL_DEFENDER_BASE:-/path/to/defender-base-model}"
MODEL_ATTACKER_SFT="${MODEL_ATTACKER_SFT:-/path/to/attacker-sft-model}"

# Switch logic
ATTACKER_RATIO="${ATTACKER_RATIO:-1}"
DEFENDER_RATIO="${DEFENDER_RATIO:-1}"
UPDATE_FREQ="${UPDATE_FREQ:-15}"
SWITCH_MODE="${SWITCH_MODE:-ratio}"  # metric | ratio
SWITCH_METRIC_NAME="${SWITCH_METRIC_NAME:-reward/response_harm}"
SWITCH_METRIC_LOW="${SWITCH_METRIC_LOW:-0.80}"
SWITCH_METRIC_HIGH="${SWITCH_METRIC_HIGH:-0.90}"
SWITCH_METRIC_WINDOW="${SWITCH_METRIC_WINDOW:-3}"

############################
# 2) Derived names
############################

timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
project_name=game
experiment_name="D-q257bi-A-q257bi-${timestamp}"

############################
# 3) Bootstrap
############################

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src/verl:${PYTHONPATH:-}"

ray stop --force || true
mkdir -p "logs/${project_name}/${experiment_name}"
ray start --head --port="$RAY_MASTER_PORT" --dashboard-host=0.0.0.0 --num-gpus 8
sleep 30

############################
# 4) Switch args
############################

SWITCH_ARGS=()
if [[ "${SWITCH_MODE}" == "metric" ]]; then
  SWITCH_ARGS=(
    "algorithm.switch_agent.mode=metric"
    "+algorithm.switch_agent.metric_name=${SWITCH_METRIC_NAME}"
    "+algorithm.switch_agent.metric_low=${SWITCH_METRIC_LOW}"
    "+algorithm.switch_agent.metric_high=${SWITCH_METRIC_HIGH}"
    "+algorithm.switch_agent.metric_window=${SWITCH_METRIC_WINDOW}"
  )
else
  SWITCH_ARGS=(
    "algorithm.switch_agent.mode=ratio"
    "algorithm.switch_agent.freq=${UPDATE_FREQ}"
    "algorithm.switch_agent.update_ratio={}"
    "+algorithm.switch_agent.update_ratio.attacker=${ATTACKER_RATIO}"
    "+algorithm.switch_agent.update_ratio.defender=${DEFENDER_RATIO}"
  )
fi

############################
# 5) Train
############################

PYTHONUNBUFFERED=1 python -m verl.separated_trainer.main_ppo \
  trainer.project_name=Game-separated \
  trainer.experiment_name="${experiment_name}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}/Game-separated/${experiment_name}" \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.val_batch_size=256 \
  data.train_batch_size=64 \
  data.max_prompt_length=8192 \
  data.max_response_length=6144 \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=1e-3 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
  actor_rollout_ref.rollout.max_num_turns=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.stop_when_truncated=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.stop_when_truncated_roles=[defender] \
  +trainer.val_before_train=False \
  +trainer.val_only=False \
  +trainer.save_val_generations=True \
  +trainer.save_train_generations=True \
  trainer.test_freq=15 \
  trainer.save_freq=15 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=300 \
  algorithm.adv_estimator=grpo \
  algorithm.switch_agent.model_paths=[${QWEN_257BI_MODEL_PATH},${QWEN_257BI_MODEL_PATH}] \
  algorithm.switch_agent.agent_roles=[attacker,defender] \
  algorithm.switch_agent.train_roles=[attacker,defender] \
  algorithm.switch_agent.start_agent=defender \
  "${SWITCH_ARGS[@]}" \
  reward_model.reward_manager=game \
  reward_model.mask_unfinished_reward=True \
  +reward_model.use_format_reward=True \
  +reward_model.format_reward_roles=[attacker] \
  algorithm.filter_groups.enable=False \
  trainer.logger=[console,wandb]

ray stop --force || true
