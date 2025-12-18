#!/bin/bash
# BERTScore Evaluation Script
# Compare base model vs fine-tuned model

set -e

# Activate environment
source ../venv/bin/activate

# CUDA setup
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL="google/gemma-3-1b-it"
FINETUNED_PATH="./outputs/gemma3-1b-qlora-sft/final_model"
TEST_DATASET="../data/final/multiturn_dataset_cleaned.json"
OUTPUT="./outputs/evaluation_results.json"

# Batch processing for speed
BATCH_SIZE=16
MAX_SAMPLES=100  # Number of samples to evaluate

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo "============================================================"
echo "BERTScore Evaluation: Base vs Fine-tuned Model"
echo "============================================================"
echo "Base Model: $BASE_MODEL"
echo "Fine-tuned: $FINETUNED_PATH"
echo "Test Samples: $MAX_SAMPLES"
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
echo "Results saved to: $OUTPUT"
