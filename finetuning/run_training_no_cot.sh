#!/bin/bash
# Training QLoRA SFT WITHOUT Chain-of-Thought (No Thought Tags)
# 3 Epochs

set -e

source ./venv_finetuning/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# CONFIGURATION - NO THOUGHT/COT
# =============================================================================

MODEL_NAME="google/gemma-3-1b-it"

# Dataset WITHOUT thought tags (from data/final/split)
TRAIN_DATASET="../data/final/split/merged_all_categories_train_no_cot.json"
EVAL_DATASET="../data/final/split/merged_all_categories_eval_no_cot.json"

# Output directory for NO-COT model
OUTPUT_DIR="./outputs/gemma3-1b-qlora-no-cot"

# Training params
EPOCHS=3
BATCH_SIZE=16
GRAD_ACCUM=4
LEARNING_RATE=2e-4
LORA_R=32
LORA_ALPHA=64
MAX_SEQ_LENGTH=2048

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "============================================================"
echo "Training QLoRA SFT - NO Chain-of-Thought"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Train: $TRAIN_DATASET"
echo "Eval:  $EVAL_DATASET"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch:  $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "============================================================"

python train_qlora_sft.py \
    --dataset "$TRAIN_DATASET" \
    --eval_dataset "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LEARNING_RATE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --max_seq_length $MAX_SEQ_LENGTH

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "============================================================"
