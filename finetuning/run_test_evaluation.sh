#!/bin/bash
# Test Evaluation Script
# Compare base model vs fine-tuned model on test data
# Output: predictions + expected answers for BERTScore/LLM-as-judge

set -e

# Activate environment
source ./venv_finetuning/bin/activate

# CUDA setup
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL="google/gemma-3-1b-it"
FINETUNED_PATH="./outputs/gemma3-1b-qlora-no-cot/final_model"
TEST_DATASET="../data/final/split/merged_all_categories_test.json"
OUTPUT="./outputs/test_evaluation_results.json"

# Batch processing for speed
BATCH_SIZE=32
MAX_SAMPLES=0  # 0 = use ALL samples

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo "============================================================"
echo "Test Evaluation: Base vs Fine-tuned Model"
echo "============================================================"
echo "Base Model: $BASE_MODEL"
echo "Fine-tuned: $FINETUNED_PATH"
echo "Test Data:  $TEST_DATASET"
echo "Output:     $OUTPUT"
echo "Max Samples: $MAX_SAMPLES (0 = ALL)"
echo "Batch Size: $BATCH_SIZE"
echo "============================================================"

python evaluate_bertscore.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --test_dataset "$TEST_DATASET" \
    --output "$OUTPUT" \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT"
echo "============================================================"
echo ""
echo "Output contains:"
echo "  - Base model responses + BERTScore"
echo "  - Fine-tuned model responses + BERTScore"  
echo "  - Comparison metrics"
echo ""
echo "You can use this output for LLM-as-judge evaluation next."
