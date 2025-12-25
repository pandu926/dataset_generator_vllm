#!/bin/bash
# Evaluation Script: Generate responses once, then evaluate with BERTScore and LLM-as-Judge
# Using the pre-trained R32 model from outputs/gemma3-lora-r32-a64-e4

set -e

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

source /workspace/dataset_generator_vllm/finetuning/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false

cd /workspace/dataset_generator_vllm/finetuning

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL="google/gemma-3-1b-it"
FINETUNED_PATH="./outputs/gemma3-lora-r32-a64-e4/final_model"
TEST_DATA="../data/final/split/merged_all_categories_test.json"
OUTPUT_DIR="./outputs/evaluation_r32"

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "EVALUATION PIPELINE: Generate Responses + BERTScore + LLM-Judge"
echo "============================================================"
echo "Base Model: $BASE_MODEL"
echo "Finetuned: $FINETUNED_PATH"
echo "Test Data: $TEST_DATA"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# =============================================================================
# STEP 1: BERTScore Evaluation (generates responses internally)
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 1: BERTScore Evaluation"
echo "============================================================"

python3 evaluate_bertscore.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --test_dataset "$TEST_DATA" \
    --output "$OUTPUT_DIR/bertscore_results.json" \
    --batch_size 16 \
    --max_samples 0

echo "BERTScore evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/bertscore_results.json"

# =============================================================================
# STEP 2: LLM-as-Judge Evaluation
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 2: LLM-as-Judge Evaluation"
echo "============================================================"

python3 evaluate_llm_judge.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --judge_model "google/gemma-3-12b-it" \
    --test_dataset "$TEST_DATA" \
    --output "$OUTPUT_DIR/llm_judge_results.json" \
    --generation_batch_size 16 \
    --judge_batch_size 32 \
    --max_samples 0

echo "LLM-as-Judge evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/llm_judge_results.json"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - BERTScore: $OUTPUT_DIR/bertscore_results.json"
echo "  - LLM Judge: $OUTPUT_DIR/llm_judge_results.json"
echo "============================================================"
