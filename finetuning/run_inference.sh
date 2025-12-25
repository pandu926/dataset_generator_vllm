#!/bin/bash
# Single-turn Inference - No CoT Model
# First turn only

set -e

source ./venv_finetuning/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL="google/gemma-3-1b-it"
FINETUNED_PATH="./outputs/gemma3-1b-qlora-no-cot/final_model"
TEST_DATASET="../data/final/split/merged_all_categories_test_no_cot.json"
OUTPUT="./outputs/inference_results_single_turn.json"

BATCH_SIZE=32
MAX_SAMPLES=0

# =============================================================================
# RUN INFERENCE
# =============================================================================

echo "============================================================"
echo "Single-turn Inference: Base vs Fine-tuned (No-CoT)"
echo "============================================================"
echo "Base Model: $BASE_MODEL"
echo "Fine-tuned: $FINETUNED_PATH"
echo "Test Data:  $TEST_DATASET"
echo "Output:     $OUTPUT"
echo "============================================================"

python generate_responses.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --test_dataset "$TEST_DATASET" \
    --output "$OUTPUT" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES

echo ""
echo "Done! Output saved to: $OUTPUT"
