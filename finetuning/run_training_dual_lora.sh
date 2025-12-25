#!/bin/bash
# Training Script for Dual LoRA Configurations (Rank 32 and 64)
# Dataset: merged_all_categories (2201 samples split into train/eval/test)
# Epochs: 4
# Followed by BERTScore and LLM-as-Judge evaluation

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

# Dataset paths
TRAIN_DATA="../data/final/split/merged_all_categories_train.json"
EVAL_DATA="../data/final/split/merged_all_categories_eval.json"
TEST_DATA="../data/final/split/merged_all_categories_test.json"

# Training params
EPOCHS=4
BATCH_SIZE=8
GRAD_ACCUM=8
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=2048

echo "============================================================"
echo "TRAINING PIPELINE: Dual LoRA Configurations"
echo "============================================================"
echo "Train: $TRAIN_DATA (1650 samples)"
echo "Eval: $EVAL_DATA (330 samples)"
echo "Test: $TEST_DATA (221 samples)"
echo "Epochs: $EPOCHS"
echo "============================================================"

# =============================================================================
# TRAINING 1: LoRA Rank 32, Alpha 64
# =============================================================================

echo ""
echo "============================================================"
echo "TRAINING 1: LoRA Rank 32, Alpha 64"
echo "============================================================"

OUTPUT_DIR_R32="./outputs/gemma3-lora-r32-a64-e4"

python3 train_qlora_sft.py \
    --dataset "$TRAIN_DATA" \
    --eval_dataset "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR_R32" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LEARNING_RATE \
    --lora_r 32 \
    --lora_alpha 64 \
    --max_seq_length $MAX_SEQ_LENGTH

echo "LoRA R32 training complete! Model saved to: $OUTPUT_DIR_R32/final_model"

# =============================================================================
# TRAINING 2: LoRA Rank 64, Alpha 128
# =============================================================================

echo ""
echo "============================================================"
echo "TRAINING 2: LoRA Rank 64, Alpha 128"
echo "============================================================"

OUTPUT_DIR_R64="./outputs/gemma3-lora-r64-a128-e4"

python3 train_qlora_sft.py \
    --dataset "$TRAIN_DATA" \
    --eval_dataset "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR_R64" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LEARNING_RATE \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_seq_length $MAX_SEQ_LENGTH

echo "LoRA R64 training complete! Model saved to: $OUTPUT_DIR_R64/final_model"

# =============================================================================
# EVALUATION 1: BERTScore for LoRA R32
# =============================================================================

echo ""
echo "============================================================"
echo "EVALUATION: BERTScore for LoRA R32"
echo "============================================================"

python3 evaluate_bertscore.py \
    --base_model "google/gemma-3-1b-it" \
    --finetuned_path "$OUTPUT_DIR_R32/final_model" \
    --test_dataset "$TEST_DATA" \
    --output "$OUTPUT_DIR_R32/bertscore_results.json" \
    --batch_size 16 \
    --max_samples 0

# =============================================================================
# EVALUATION 2: BERTScore for LoRA R64
# =============================================================================

echo ""
echo "============================================================"
echo "EVALUATION: BERTScore for LoRA R64"
echo "============================================================"

python3 evaluate_bertscore.py \
    --base_model "google/gemma-3-1b-it" \
    --finetuned_path "$OUTPUT_DIR_R64/final_model" \
    --test_dataset "$TEST_DATA" \
    --output "$OUTPUT_DIR_R64/bertscore_results.json" \
    --batch_size 16 \
    --max_samples 0

# =============================================================================
# EVALUATION 3: LLM-as-Judge for LoRA R32
# =============================================================================

echo ""
echo "============================================================"
echo "EVALUATION: LLM-as-Judge for LoRA R32"
echo "============================================================"

python3 evaluate_llm_judge.py \
    --base_model "google/gemma-3-1b-it" \
    --finetuned_path "$OUTPUT_DIR_R32/final_model" \
    --judge_model "google/gemma-3-12b-it" \
    --test_dataset "$TEST_DATA" \
    --output "$OUTPUT_DIR_R32/llm_judge_results.json" \
    --generation_batch_size 16 \
    --judge_batch_size 32 \
    --max_samples 0

# =============================================================================
# EVALUATION 4: LLM-as-Judge for LoRA R64
# =============================================================================

echo ""
echo "============================================================"
echo "EVALUATION: LLM-as-Judge for LoRA R64"
echo "============================================================"

python3 evaluate_llm_judge.py \
    --base_model "google/gemma-3-1b-it" \
    --finetuned_path "$OUTPUT_DIR_R64/final_model" \
    --judge_model "google/gemma-3-12b-it" \
    --test_dataset "$TEST_DATA" \
    --output "$OUTPUT_DIR_R64/llm_judge_results.json" \
    --generation_batch_size 16 \
    --judge_batch_size 32 \
    --max_samples 0

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "ALL TRAINING AND EVALUATION COMPLETE!"
echo "============================================================"
echo ""
echo "Models:"
echo "  - LoRA R32: $OUTPUT_DIR_R32/final_model"
echo "  - LoRA R64: $OUTPUT_DIR_R64/final_model"
echo ""
echo "Evaluation Results:"
echo "  - LoRA R32 BERTScore: $OUTPUT_DIR_R32/bertscore_results.json"
echo "  - LoRA R32 LLM Judge: $OUTPUT_DIR_R32/llm_judge_results.json"
echo "  - LoRA R64 BERTScore: $OUTPUT_DIR_R64/bertscore_results.json"
echo "  - LoRA R64 LLM Judge: $OUTPUT_DIR_R64/llm_judge_results.json"
echo "============================================================"
