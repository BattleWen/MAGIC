#!/bin/bash

# ============================================================================
# RL Data Classification - Fine-Grained Attack Strategy Classification
# ============================================================================

# Classify model API Configuration
export API_KEY="$API_KEY"
export API_URL="$API_URL"
export MODEL_NAME="$MOMEL_NAME"

# Path Configuration
EXTRACTED_DIR="/mnt/shared-storage-user/wenxiaoyu/MAGIC/eval/pattern/raw_json"
MODEL_PATH=
OUTPUT_DIR="/mnt/shared-storage-user/wenxiaoyu/MAGIC/eval/pattern/results/rl_grained"

# Log Configuration
LOG_DIR="$LOG_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/rl_classification_${TIMESTAMP}.log"

# ============================================================================
# Print Information
# ============================================================================

echo "================================================================================"
echo "RL Data Fine-Grained Classification - Started"
echo "================================================================================"
echo "Start time: $(date)"
echo "API URL: $API_URL"
echo "Model: $MODEL_NAME"
echo "Extracted directory: $EXTRACTED_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Step range: $STEP_START to $STEP_END (interval: $STEP_INTERVAL)"
echo "Process all lines: $PROCESS_ALL_LINES"
echo "Log file: $LOG_FILE"
echo "================================================================================"
echo ""

# ============================================================================
# Run Classification
# ============================================================================

python classify_attacker_prompts.py \
    --mode rl \
    --use-grained \
    --api-key "$API_KEY" \
    --api-url "$API_URL" \
    --model "$MODEL_NAME" \
    --extracted-dir "$EXTRACTED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $(if [ "$PROCESS_ALL_LINES" = true ]; then echo "--process-all-lines"; fi) \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "================================================================================"
echo "Processing completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log file: $LOG_FILE"
echo "================================================================================"

exit $EXIT_CODE
