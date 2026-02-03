#!/usr/bin/env bash
set -euo pipefail

# Unified Evaluation Process (Template Script)
# - API Evaluation: StrongREJECT / AlpacaEval-2.0 (It is recommended to use GPT-4o for scoring)
# - vLLM Evaluation: Safety benchmark TASKS + MMLU (GPU)
# - X-Teaming: One-click generation/attack/statistics

# =============================
# Required/Optional Configuration (Please replace placeholders)
# =============================

# Defender API (Generate responses)
DEFENDER_API_BASE_URL="${DEFENDER_API_BASE_URL:-}"
DEFENDER_API_KEY="${DEFENDER_API_KEY:-}"
DEFENDER_API_MODEL="${DEFENDER_API_MODEL:-}"
DEFENDER_MODEL_NAME="${DEFENDER_MODEL_NAME:-}"

# OpenAI Scoring API (For AlpacaEval-2.0 / StrongREJECT)
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OPENAI_MODEL="${OPENAI_MODEL:-}"

# Guard API Qwen3guard/Wildguard (Optional, for security evaluation)
WILDGUARD_API_ENDPOINT="${WILDGUARD_API_ENDPOINT:-}"
WILDGUARD_API_KEY="${WILDGUARD_API_KEY:-}"
CLASSIFIER_MODEL_NAME="${CLASSIFIER_MODEL_NAME:-Qwen3GuardAPI}"

# vLLM Model and Template
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
# Qwen uses game_defender; Llama uses hf
MODEL_TEMPLATE="${MODEL_TEMPLATE:-game_defender}"
MMLU_TEMPLATE="${MMLU_TEMPLATE:-hf}"

# Task List (Security tasks + MMLU)
TASKS="wildguardtest,harmbench_precompute,wildjailbreak:benign,wildjailbreak:harmful,do_anything_now,harmbench,or_bench:hard-1k,or_bench:toxic,xstest"

# Results Directory
RESULTS_ROOT="${RESULTS_ROOT:-./results/unified}"

# Hugging Face
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_cache}"
HF_TOKEN="${HF_TOKEN:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_DATASETS_VERBOSITY="${HF_DATASETS_VERBOSITY:-info}"

# =============================
# Optional: Environment Activation
# =============================
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate safety-eval || true
fi

# =============================
# Utility Functions
# =============================
warn_if_empty() {
  local name="$1"
  local value="$2"
  if [ -z "$value" ]; then
    echo "[WARN] Variable $name is not set."
  fi
}

print_api_config() {
  echo "[INFO] Defender API: ${DEFENDER_API_BASE_URL:-<empty>}"
  echo "[INFO] OpenAI API: ${OPENAI_BASE_URL:-<empty>}"
}

# =============================
# 1) API Evaluation: StrongREJECT / AlpacaEval-2.0
# =============================
run_api_eval() {
  echo "===== API Evaluation Start ====="
  warn_if_empty DEFENDER_API_BASE_URL "$DEFENDER_API_BASE_URL"
  warn_if_empty DEFENDER_API_KEY "$DEFENDER_API_KEY"
  warn_if_empty DEFENDER_API_MODEL "$DEFENDER_API_MODEL"
  warn_if_empty DEFENDER_MODEL_NAME "$DEFENDER_MODEL_NAME"
  warn_if_empty OPENAI_API_KEY "$OPENAI_API_KEY"
  warn_if_empty OPENAI_BASE_URL "$OPENAI_BASE_URL"
  warn_if_empty OPENAI_MODEL "$OPENAI_MODEL"
  print_api_config

  export DEFENDER_API_BASE_URL DEFENDER_API_KEY DEFENDER_API_MODEL
  export OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL

  local ts
  ts=$(date +%Y%m%d_%H%M%S)

  # AlpacaEval-2.0
  local alpaca_dir="$RESULTS_ROOT/api/alpacaeval-$ts"
  mkdir -p "$alpaca_dir"
  timeout 7200 python -u evaluation/eval.py generators \
    --model_name_or_path "$DEFENDER_MODEL_NAME" \
    --model_input_template_path_or_name "hf" \
    --tasks "alpacaeval" \
    --report_output_path "$alpaca_dir/metrics.json" \
    --save_individual_results_path "$alpaca_dir/all_results.json" \
    --use_defender_api \
    --batch_size 1 2>&1 | tee "$alpaca_dir/eval.log"

  # StrongREJECT
  local strongreject_dir="$RESULTS_ROOT/api/strongreject-$ts"
  mkdir -p "$strongreject_dir"
  timeout 7200 python -u evaluation/eval.py generators \
    --model_name_or_path "$DEFENDER_MODEL_NAME" \
    --model_input_template_path_or_name "hf" \
    --tasks "strongreject" \
    --report_output_path "$strongreject_dir/metrics.json" \
    --save_individual_results_path "$strongreject_dir/all_results.json" \
    --use_defender_api \
    --use_gpt4_autograder \
    --batch_size 1 2>&1 | tee "$strongreject_dir/eval.log"

  echo "===== API Evaluation Complete ====="
}

# =============================
# 2) vLLM Evaluation: Security Tasks + MMLU
# =============================
run_vllm_eval() {
  echo "===== vLLM Evaluation Start ====="
  warn_if_empty MODEL_PATH "$MODEL_PATH"
  warn_if_empty MODEL_NAME "$MODEL_NAME"
  warn_if_empty WILDGUARD_API_ENDPOINT "$WILDGUARD_API_ENDPOINT"

  export WILDGUARD_API_ENDPOINT WILDGUARD_API_KEY
  export HF_DATASETS_CACHE HF_TOKEN HF_HUB_OFFLINE HF_DATASETS_VERBOSITY

  local ts
  ts=$(date +%Y%m%d_%H%M%S)

  # Security Tasks (TASKS)
  local safety_dir="$RESULTS_ROOT/vllm/safety-$MODEL_NAME-$ts"
  mkdir -p "$safety_dir"
  timeout 10800 python -u evaluation/eval.py generators \
    --model_name_or_path "$MODEL_PATH" \
    --model_input_template_path_or_name "$MODEL_TEMPLATE" \
    --tasks "$TASKS" \
    --report_output_path "$safety_dir/metrics.json" \
    --save_individual_results_path "$safety_dir/all_results.json" \
    --use_vllm \
    --classifier_model_name "$CLASSIFIER_MODEL_NAME" \
    --no_extract_answer false \
    --batch_size 1 2>&1 | tee "$safety_dir/eval.log"

  # MMLU (General Capabilities)
  local mmlu_dir="$RESULTS_ROOT/vllm/mmlu-$MODEL_NAME-$ts"
  mkdir -p "$mmlu_dir"
  timeout 7200 python -u evaluation/eval.py generators \
    --model_name_or_path "$MODEL_PATH" \
    --model_input_template_path_or_name "$MMLU_TEMPLATE" \
    --tasks "mmlu" \
    --report_output_path "$mmlu_dir/metrics.json" \
    --save_individual_results_path "$mmlu_dir/all_results.json" \
    --use_vllm \
    --batch_size 1 2>&1 | tee "$mmlu_dir/eval.log"

  echo "===== vLLM Evaluation Complete ====="
}

# =============================
# 3) X-Teaming Evaluation
# =============================
run_xteaming_eval() {
  echo "===== X-Teaming Evaluation Start ====="
  python run_xteaming_benchmark.py --stage all
  echo "===== X-Teaming Evaluation Complete ====="
}

# =============================
# Entry Point
# =============================
main() {
  run_api_eval
  run_vllm_eval
  run_xteaming_eval
}

main "$@"