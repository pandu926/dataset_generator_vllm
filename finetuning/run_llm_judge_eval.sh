#!/bin/bash
# LLM-as-Judge - Generate from scratch + RAG grounding
# Single-turn evaluation with No-CoT model

set -e

source ./venv_finetuning/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL="google/gemma-3-1b-it"
FINETUNED_PATH="./outputs/gemma3-1b-qlora-no-cot/final_model"
JUDGE_MODEL="google/gemma-3-12b-it"

TEST_DATASET="../data/final/split/merged_all_categories_test_no_cot.json"
OUTPUT="./outputs/llm_judge_results.json"

GEN_BATCH_SIZE=16
JUDGE_BATCH_SIZE=32
MAX_SAMPLES=0  # 0 = ALL

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo "============================================================"
echo "LLM-as-Judge: Generate from scratch + RAG grounding"
echo "============================================================"
echo "Base Model:  $BASE_MODEL"
echo "Fine-tuned:  $FINETUNED_PATH"
echo "Judge Model: $JUDGE_MODEL"
echo "Test Data:   $TEST_DATASET"
echo "Max Samples: $MAX_SAMPLES (0 = ALL)"
echo "============================================================"

python evaluate_llm_judge.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --judge_model "$JUDGE_MODEL" \
    --test_dataset "$TEST_DATASET" \
    --output "$OUTPUT" \
    --gen_batch_size $GEN_BATCH_SIZE \
    --judge_batch_size $JUDGE_BATCH_SIZE \
    --max_samples $MAX_SAMPLES

echo ""
echo "============================================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT"
echo "============================================================"
