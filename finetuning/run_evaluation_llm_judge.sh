#!/bin/bash
# LLM-as-Judge Evaluation Script
# Compare base vs fine-tuned model using Gemma-3-12B judge
# Optimized for A100 80GB

set -e  # Exit on error

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate virtual environment
source ../venv/bin/activate

# Set CUDA
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# CONFIGURATION
# =============================================================================

# Models
BASE_MODEL="google/gemma-3-1b-it"
FINETUNED_PATH="./outputs/gemma3-1b-qlora-sft/final_model"
JUDGE_MODEL="google/gemma-3-12b-it"

# Dataset (without thought tags)
TEST_DATASET="../data/final/multiturn_dataset_cleaned_no_thought.json"
OUTPUT_PATH="./outputs/llm_judge_evaluation_results.json"

# Batch sizes (optimized for A100 80GB)
GEN_BATCH_SIZE=16    # For generating responses
JUDGE_BATCH_SIZE=32  # For judging responses

# Number of samples to evaluate
MAX_SAMPLES=250

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo "============================================================"
echo "LLM-as-Judge Evaluation: Base vs Fine-tuned Model"
echo "============================================================"
echo "Judge Model: $JUDGE_MODEL"
echo "Base Model: $BASE_MODEL"  
echo "Fine-tuned: $FINETUNED_PATH"
echo "Test Samples: $MAX_SAMPLES"
echo "============================================================"
echo ""
echo "5-Point Likert Scale Dimensions:"
echo "  1. Helpfulness (1-5)"
echo "  2. Relevance (1-5)"
echo "  3. Accuracy (1-5)"
echo "  4. Coherence (1-5)"
echo "  5. Fluency (1-5)"
echo "============================================================"

python evaluate_llm_judge.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --judge_model "$JUDGE_MODEL" \
    --test_dataset "$TEST_DATASET" \
    --output "$OUTPUT_PATH" \
    --gen_batch_size $GEN_BATCH_SIZE \
    --judge_batch_size $JUDGE_BATCH_SIZE \
    --max_samples $MAX_SAMPLES

echo ""
echo "============================================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "============================================================"
