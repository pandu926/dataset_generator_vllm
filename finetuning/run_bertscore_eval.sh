#!/bin/bash
# BERTScore Evaluation from Inference Results

set -e

source ./venv_finetuning/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

INPUT="./outputs/inference_results_multiturn.json"
OUTPUT="./outputs/bertscore_results.json"

echo "============================================================"
echo "BERTScore Evaluation from Inference Results"
echo "============================================================"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "============================================================"

python evaluate_from_inference.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --batch_size 32

echo ""
echo "Done! Results saved to: $OUTPUT"
