#!/bin/bash
# =============================================================================
# Run Full RAG-Augmented LLM-as-Judge Evaluation
# Both base and fine-tuned models receive RAG context during generation
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$(dirname "$SCRIPT_DIR")/venv"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_DIR"
    echo "Using system Python"
fi

# Default parameters
BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
FINETUNED_PATH="${FINETUNED_PATH:-./outputs/gemma3-1b-qlora-sft/final_model}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-3-12b-it}"
TEST_DATASET="${TEST_DATASET:-../data/split/multiturn_dataset_cleaned_no_thought_test.json}"
CHUNKS_PATH="${CHUNKS_PATH:-../data/chunks/chunks.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-./outputs/llm_judge_rag_evaluation_results.json}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
RAG_TOP_K="${RAG_TOP_K:-3}"

echo "============================================================"
echo "FULL RAG-AUGMENTED LLM-AS-JUDGE EVALUATION"
echo "============================================================"
echo "Base Model:      $BASE_MODEL"
echo "Fine-tuned Path: $FINETUNED_PATH"
echo "Judge Model:     $JUDGE_MODEL"
echo "Test Dataset:    $TEST_DATASET"
echo "RAG Chunks:      $CHUNKS_PATH"
echo "Output:          $OUTPUT_PATH"
echo "Max Samples:     $MAX_SAMPLES"
echo "RAG Top-K:       $RAG_TOP_K"
echo "============================================================"
echo ""
echo "KEY DIFFERENCE: Both models will receive RAG context during generation!"
echo ""

# Run evaluation
cd "$SCRIPT_DIR"

python evaluate_llm_judge_rag.py \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FINETUNED_PATH" \
    --judge_model "$JUDGE_MODEL" \
    --test_dataset "$TEST_DATASET" \
    --chunks_path "$CHUNKS_PATH" \
    --output "$OUTPUT_PATH" \
    --max_samples "$MAX_SAMPLES" \
    --rag_top_k "$RAG_TOP_K"

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "============================================================"
