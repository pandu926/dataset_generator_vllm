#!/bin/bash
# QLoRA SFT Training Script for Gemma 3-1B
# Configuration: LoRA r=32, alpha=64, lr=2e-4, epochs=3

set -e  # Exit on error

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate virtual environment (skip if already active)
# source ../venv/bin/activate

# Set CUDA
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset - menggunakan dataset yang sudah di-split
TRAIN_DATASET="../data/sudah_bagus/train.json"
EVAL_DATASET="../data/sudah_bagus/eval.json"
OUTPUT_DIR="./outputs/gemma3-1b-r32-a64-lr2e4-e3"

# Training params
EPOCHS=3
BATCH_SIZE=8
GRAD_ACCUM=8
LEARNING_RATE=2e-4

# LoRA params
LORA_R=32
LORA_ALPHA=64

# Sequence length
MAX_SEQ_LENGTH=2048

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "============================================================"
echo "Starting QLoRA SFT Training for Gemma 3-1B"
echo "============================================================"
echo "Train Dataset: $TRAIN_DATASET"
echo "Eval Dataset: $EVAL_DATASET"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Learning Rate: $LEARNING_RATE"
echo "LoRA rank: $LORA_R, alpha: $LORA_ALPHA"
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
