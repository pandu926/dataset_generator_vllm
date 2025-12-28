#!/bin/bash
# =============================================================================
# Run Batch LLM-as-Judge Evaluation with OpenAI GPT-5 Mini
# Uses async parallel requests for maximum speed
# =============================================================================

# Exit on error
set -e

echo "=========================================="
echo "Batch LLM-as-Judge with OpenAI GPT-5 Mini"
echo "=========================================="

# Navigate to finetuning directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv_finetuning" ]; then
    echo "Activating venv_finetuning..."
    source venv_finetuning/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating ../venv..."
    source ../venv/bin/activate
else
    echo "Warning: No virtual environment found, using system Python"
fi

# Install openai if not present
pip install openai tqdm --quiet

# =============================================================================
# CONFIGURATION
# =============================================================================

# Concurrent OpenAI requests (50 is safe, can go up to 100)
CONCURRENT_REQUESTS=50

# Generation batch size for local model
GEN_BATCH_SIZE=32

# Number of RAG chunks to retrieve
RAG_TOP_K=3

# Max samples (0 = ALL)
MAX_SAMPLES=0

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo ""
echo "Configuration:"
echo "  Concurrent OpenAI requests: $CONCURRENT_REQUESTS"
echo "  Generation batch size: $GEN_BATCH_SIZE"
echo "  RAG top-k: $RAG_TOP_K"
echo "  Max samples: $MAX_SAMPLES (0=all)"
echo ""

python batch_llm_judge_openai.py \
    --concurrent $CONCURRENT_REQUESTS \
    --gen_batch $GEN_BATCH_SIZE \
    --rag_top_k $RAG_TOP_K \
    --max_samples $MAX_SAMPLES

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: outputs/batch_evaluation_results/"
echo "=========================================="
