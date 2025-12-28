#!/bin/bash
# =============================================================================
# Batch LLM-as-Judge Evaluation for All Fine-tuned Models
# Evaluates all models in model_hasil_reserach_parameter folder
# Using Gemma-3-12B as judge WITH RAG grounding
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Batch LLM-as-Judge Evaluation"
echo "All Fine-tuned Models with RAG"
echo "=============================================="
echo ""
echo "Working directory: $SCRIPT_DIR"
echo ""

# Configuration
BASE_MODEL="google/gemma-3-1b-it"
JUDGE_MODEL="google/gemma-3-12b-it"
MODELS_DIR="./model_hasil_reserach_parameter"
TEST_DATASET="../data/final/split/merged_all_categories_test_no_cot.json"
OUTPUT_DIR="./outputs/batch_evaluation_results"
CHUNKS_PATH="../data/chunks/chunks.jsonl"

# Batch sizes - adjust based on your GPU memory
GEN_BATCH_SIZE=32
JUDGE_BATCH_SIZE=32

# Number of test samples (0 = use ALL samples)
MAX_SAMPLES=0

# RAG settings
RAG_TOP_K=3

# Activate virtual environment
if [ -d "./venv_finetuning" ]; then
    echo "Activating venv_finetuning..."
    source ./venv_finetuning/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating ../venv..."
    source ../venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Check if required files exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory not found: $MODELS_DIR"
    exit 1
fi

if [ ! -f "$TEST_DATASET" ]; then
    echo "ERROR: Test dataset not found: $TEST_DATASET"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo ""
echo "Configuration:"
echo "  Base Model:      $BASE_MODEL"
echo "  Judge Model:     $JUDGE_MODEL"
echo "  Models Dir:      $MODELS_DIR"
echo "  Test Dataset:    $TEST_DATASET"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Chunks Path:     $CHUNKS_PATH"
echo "  Gen Batch Size:  $GEN_BATCH_SIZE"
echo "  Judge Batch:     $JUDGE_BATCH_SIZE"
echo "  Max Samples:     $MAX_SAMPLES (0 = all)"
echo "  RAG Top-K:       $RAG_TOP_K"
echo ""

# List models to evaluate
echo "Models to evaluate:"
for model_dir in "$MODELS_DIR"/*; do
    if [ -d "$model_dir/final_model" ]; then
        echo "  - $(basename $model_dir)"
    fi
done
echo ""

# Confirm before starting (optional - comment out for unattended runs)
# read -p "Press Enter to start evaluation or Ctrl+C to cancel..."

# Run batch evaluation
echo "Starting batch evaluation..."
echo ""

python batch_llm_judge_finetuned_only.py \
    --base_model "$BASE_MODEL" \
    --judge_model "$JUDGE_MODEL" \
    --models_dir "$MODELS_DIR" \
    --test_dataset "$TEST_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --chunks_path "$CHUNKS_PATH" \
    --gen_batch_size $GEN_BATCH_SIZE \
    --judge_batch_size $JUDGE_BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --rag_top_k $RAG_TOP_K

echo ""
echo "=============================================="
echo "Batch evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
