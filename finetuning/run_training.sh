#!/bin/bash
# QLoRA SFT Training Script for Gemma 3-1B
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

# Dataset
DATASET="../data/final/multiturn_dataset_cleaned_no_thought.json"
OUTPUT_DIR="./outputs/gemma3-1b-qlora-sft"

# Training params (Best Practices 2024)
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
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "LoRA rank: $LORA_R, alpha: $LORA_ALPHA"
echo "============================================================"

python train_qlora_sft.py \
    --dataset "$DATASET" \
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
