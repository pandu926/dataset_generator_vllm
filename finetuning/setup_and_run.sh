#!/bin/bash
# =============================================================================
# COMPLETE SETUP & RUN SCRIPT
# QLoRA SFT Fine-tuning for Gemma 3-1B-IT
# With Dataset Split and LLM-as-Judge Evaluation
# =============================================================================
# 
# This script handles everything:
# 1. Virtual environment setup
# 2. Dependencies installation (including vLLM)
# 3. Dataset preparation (split into train/eval/test)
# 4. Model training with QLoRA
# 5. Evaluation with LLM-as-Judge (5-point Likert scale)
#
# Usage: ./setup_and_run.sh [OPTIONS]
#   --setup-only     Only setup environment, don't train
#   --train-only     Skip setup, only train
#   --eval-only      Skip training, only evaluate
#   --split-only     Only split dataset
#   --all            Run complete pipeline (default)
#   --help           Show this help message
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNING_DIR="$PROJECT_ROOT/finetuning"
VENV_DIR="$FINETUNING_DIR/venv_finetuning"  # New dedicated venv for finetuning

# Dataset Configuration - Updated for merged dataset
DATASET_SOURCE="$PROJECT_ROOT/data/merged/multiturn_merged_clean.json"
DATASET_DIR="$PROJECT_ROOT/data/merged"
TRAIN_DATASET="$DATASET_DIR/multiturn_train.json"
EVAL_DATASET="$DATASET_DIR/multiturn_eval.json"
TEST_DATASET="$DATASET_DIR/multiturn_test.json"

OUTPUT_DIR="$FINETUNING_DIR/outputs/gemma3-1b-qlora-sft"

# CUDA
CUDA_VERSION="12.6"
CUDA_PATH="/usr/local/cuda-$CUDA_VERSION"

# Training Configuration (can be overridden via environment variables)
MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-3-12b-it}"
EPOCHS="${EPOCHS:-4}"                   # 4 epochs
BATCH_SIZE="${BATCH_SIZE:-16}"          # Optimized for A100 80GB
GRAD_ACCUM="${GRAD_ACCUM:-4}"           # Effective batch = 16 * 4 = 64
LEARNING_RATE="${LEARNING_RATE:-2e-4}"  # Learning rate
LORA_R="${LORA_R:-16}"                  # LoRA rank
LORA_ALPHA="${LORA_ALPHA:-32}"          # LoRA alpha (2x rank)
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# Evaluation Configuration
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"       # 0 = use ALL test samples (179 total)
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-32}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}$1${NC}"
    echo "============================================================"
}

check_cuda() {
    log_info "Checking CUDA installation..."
    if [ -d "$CUDA_PATH" ]; then
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
        log_success "CUDA $CUDA_VERSION found at $CUDA_PATH"
    else
        log_warning "CUDA $CUDA_VERSION not found at $CUDA_PATH"
        log_info "Will use system CUDA if available"
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Info:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        log_error "nvidia-smi not found. Is CUDA installed?"
        exit 1
    fi
}

check_source_dataset() {
    log_info "Checking source dataset..."
    if [ -f "$DATASET_SOURCE" ]; then
        local count=$(python3 -c "import json; print(len(json.load(open('$DATASET_SOURCE'))))")
        log_success "Dataset found: $DATASET_SOURCE ($count samples)"
    else
        log_error "Dataset not found: $DATASET_SOURCE"
        log_info "Please ensure the dataset exists"
        exit 1
    fi
}

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

setup_venv() {
    print_header "Step 1: Setting up Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        log_success "Virtual environment already exists at $VENV_DIR"
        return 0
    fi
    
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    log_success "Virtual environment created at $VENV_DIR"
}

activate_venv() {
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    log_success "Activated: $(which python3)"
}

install_dependencies() {
    print_header "Step 2: Installing Dependencies"
    
    activate_venv
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip -q
    
    # Install PyTorch with CUDA support first
    log_info "Installing PyTorch with CUDA support..."
    pip install 'torch>=2.4.0' --index-url https://download.pytorch.org/whl/cu128 -q
    
    # Install HuggingFace stack
    log_info "Installing HuggingFace stack..."
    pip install transformers accelerate peft trl bitsandbytes -q --upgrade
    
    # Install vLLM for LLM-as-Judge
    log_info "Installing vLLM (for LLM-as-Judge)..."
    pip install vllm -q
    
    log_info "Installing other dependencies..."
    pip install datasets tensorboard tqdm -q
    
    # Evaluation dependencies
    log_info "Installing evaluation dependencies..."
    pip install bert-score sentence-transformers -q
    
    # Verify installation
    log_info "Verifying installation..."
    python3 -c "
import torch
import transformers
import peft
import trl
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except:
    print('vLLM: Not installed')
"
    
    log_success "All dependencies installed!"
}

# =============================================================================
# DATASET FUNCTIONS
# =============================================================================

split_dataset() {
    print_header "Step 3: Splitting Dataset into Train/Eval/Test"
    
    activate_venv
    check_source_dataset
    
    # Check if split already exists
    if [ -f "$TRAIN_DATASET" ] && [ -f "$EVAL_DATASET" ] && [ -f "$TEST_DATASET" ]; then
        log_info "Split datasets already exist:"
        echo "  Train: $TRAIN_DATASET"
        echo "  Eval:  $EVAL_DATASET"
        echo "  Test:  $TEST_DATASET"
        
        # Show counts
        python3 -c "
import json
train = len(json.load(open('$TRAIN_DATASET')))
eval_c = len(json.load(open('$EVAL_DATASET')))
test = len(json.load(open('$TEST_DATASET')))
print(f'  Train samples: {train}')
print(f'  Eval samples:  {eval_c}')
print(f'  Test samples:  {test}')
"
        log_success "Using existing split datasets"
        return 0
    fi
    
    log_info "Splitting dataset (80% train, 10% eval, 10% test)..."
    
    cd "$FINETUNING_DIR"
    
    python3 split_dataset.py \
        --input "$DATASET_SOURCE" \
        --output_dir "$DATASET_DIR" \
        --train_ratio 0.75 \
        --eval_ratio 0.15 \
        --test_ratio 0.1 \
        --seed 42
    
    log_success "Dataset split complete!"
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

run_training() {
    print_header "Step 4: Running QLoRA SFT Training"
    
    activate_venv
    
    # Check datasets exist
    if [ ! -f "$TRAIN_DATASET" ]; then
        log_error "Training dataset not found. Run --split-only first."
        exit 1
    fi
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=0
    export TOKENIZERS_PARALLELISM=false
    
    log_info "Training Configuration:"
    echo "  Model: $MODEL_NAME"
    echo "  Train Dataset: $TRAIN_DATASET"
    echo "  Eval Dataset: $EVAL_DATASET"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
    echo "  Max Seq Length: $MAX_SEQ_LENGTH"
    echo "  Output: $OUTPUT_DIR"
    
    cd "$FINETUNING_DIR"
    
    python3 train_qlora_sft.py \
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
    
    log_success "Training completed!"
    log_info "Model saved to: $OUTPUT_DIR/final_model"
}

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

run_bertscore_evaluation() {
    print_header "Step 5a: Running BERTScore Evaluation"
    
    activate_venv
    
    # Check if fine-tuned model exists
    FINETUNED_PATH="$OUTPUT_DIR/final_model"
    if [ ! -d "$FINETUNED_PATH" ]; then
        log_error "Fine-tuned model not found at $FINETUNED_PATH"
        log_info "Please run training first"
        exit 1
    fi
    
    log_info "BERTScore Evaluation Configuration:"
    echo "  Base Model: $MODEL_NAME"
    echo "  Fine-tuned: $FINETUNED_PATH"
    echo "  Test Dataset: $TEST_DATASET"
    echo "  Test Samples: $EVAL_SAMPLES"
    
    cd "$FINETUNING_DIR"
    
    python3 evaluate_bertscore.py \
        --base_model "$MODEL_NAME" \
        --finetuned_path "$FINETUNED_PATH" \
        --test_dataset "$TEST_DATASET" \
        --output "$OUTPUT_DIR/bertscore_evaluation_results.json" \
        --batch_size $GEN_BATCH_SIZE \
        --max_samples $EVAL_SAMPLES
    
    log_success "BERTScore evaluation completed!"
    log_info "Results saved to: $OUTPUT_DIR/bertscore_evaluation_results.json"
}

run_llm_judge_evaluation() {
    print_header "Step 5b: Running LLM-as-Judge Evaluation"
    
    activate_venv
    
    # Check if fine-tuned model exists
    FINETUNED_PATH="$OUTPUT_DIR/final_model"
    if [ ! -d "$FINETUNED_PATH" ]; then
        log_error "Fine-tuned model not found at $FINETUNED_PATH"
        log_info "Please run training first"
        exit 1
    fi
    
    # Check test dataset exists
    if [ ! -f "$TEST_DATASET" ]; then
        log_error "Test dataset not found. Run --split-only first."
        exit 1
    fi
    
    log_info "LLM-as-Judge Evaluation Configuration:"
    echo "  Base Model: $MODEL_NAME"
    echo "  Fine-tuned: $FINETUNED_PATH"
    echo "  Judge Model: $JUDGE_MODEL"
    echo "  Test Dataset: $TEST_DATASET"
    echo "  Test Samples: $EVAL_SAMPLES"
    echo ""
    echo "  5-Point Likert Scale Dimensions:"
    echo "    1. Helpfulness (1-5)"
    echo "    2. Relevance (1-5)"
    echo "    3. Accuracy (1-5)"
    echo "    4. Coherence (1-5)"
    echo "    5. Fluency (1-5)"
    
    cd "$FINETUNING_DIR"
    
    python3 evaluate_llm_judge.py \
        --base_model "$MODEL_NAME" \
        --finetuned_path "$FINETUNED_PATH" \
        --judge_model "$JUDGE_MODEL" \
        --test_dataset "$TEST_DATASET" \
        --output "$OUTPUT_DIR/llm_judge_evaluation_results.json" \
        --gen_batch_size $GEN_BATCH_SIZE \
        --judge_batch_size $JUDGE_BATCH_SIZE \
        --max_samples $EVAL_SAMPLES
    
    log_success "Evaluation completed!"
    log_info "Results saved to: $OUTPUT_DIR/llm_judge_evaluation_results.json"
    
    # Display results summary
    if [ -f "$OUTPUT_DIR/llm_judge_evaluation_results.json" ]; then
        echo ""
        echo "============================================================"
        echo "EVALUATION RESULTS SUMMARY"
        echo "============================================================"
        python3 -c "
import json
with open('$OUTPUT_DIR/llm_judge_evaluation_results.json') as f:
    r = json.load(f)
base = r['base_model']['stats']
ft = r['finetuned_model']['stats']

print(f\"{'Dimension':<15} {'Base':>10} {'Finetuned':>12} {'Improvement':>12}\")
print('-'*55)
for dim in ['helpfulness', 'relevance', 'accuracy', 'coherence', 'fluency']:
    b = base[dim]['mean']
    f = ft[dim]['mean']
    imp = f - b
    print(f\"{dim.capitalize():<15} {b:>10.2f} {f:>12.2f} {imp:>+12.2f}\")
print('-'*55)
b_total = base['total']['mean']
f_total = ft['total']['mean']
pct = ((f_total - b_total) / b_total * 100) if b_total > 0 else 0
print(f\"{'OVERALL':<15} {b_total:>10.2f} {f_total:>12.2f} {f_total-b_total:>+9.2f} ({pct:+.1f}%)\")
"
    fi
}

# =============================================================================
# TENSORBOARD
# =============================================================================

run_tensorboard() {
    print_header "Starting TensorBoard"
    
    activate_venv
    
    TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
    if [ ! -d "$TENSORBOARD_DIR" ]; then
        log_error "TensorBoard logs not found at $TENSORBOARD_DIR"
        exit 1
    fi
    
    log_info "Starting TensorBoard at http://localhost:6006"
    tensorboard --logdir "$TENSORBOARD_DIR" --port 6006
}

# =============================================================================
# SHOW HELP
# =============================================================================

show_help() {
    echo "QLoRA SFT Fine-tuning Pipeline with LLM-as-Judge Evaluation"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup-only       Only setup environment (venv + dependencies)"
    echo "  --split-only       Only split dataset into train/eval/test"
    echo "  --train-only       Skip setup, only run training"
    echo "  --eval-only        Skip training, only run LLM-as-Judge evaluation"
    echo "  --tensorboard      Start TensorBoard"
    echo "  --all              Run complete pipeline (setup + split + train + eval)"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME         Model to fine-tune (default: google/gemma-3-1b-it)"
    echo "  JUDGE_MODEL        LLM-as-Judge model (default: google/gemma-3-12b-it)"
    echo "  EPOCHS             Number of training epochs (default: 3)"
    echo "  BATCH_SIZE         Per-device batch size (default: 8)"
    echo "  LEARNING_RATE      Learning rate (default: 2e-4)"
    echo "  EVAL_SAMPLES       Number of test samples to evaluate (default: 100)"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Full pipeline"
    echo "  $0 --setup-only             # Only setup"
    echo "  EPOCHS=5 $0 --train-only    # Train with 5 epochs"
    echo "  $0 --eval-only              # Only run LLM-as-Judge evaluation"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    cd "$PROJECT_ROOT"
    
    print_header "QLoRA SFT Fine-tuning Pipeline"
    echo "Project Root: $PROJECT_ROOT"
    echo "Fine-tuning Dir: $FINETUNING_DIR"
    echo "Dataset Source: $DATASET_SOURCE"
    
    check_cuda
    
    # Parse arguments
    DO_SETUP=true
    DO_SPLIT=true
    DO_TRAIN=true
    DO_EVAL=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                DO_SPLIT=false
                DO_TRAIN=false
                DO_EVAL=false
                shift
                ;;
            --split-only)
                DO_SETUP=false
                DO_TRAIN=false
                DO_EVAL=false
                shift
                ;;
            --train-only)
                DO_SETUP=false
                DO_SPLIT=false
                DO_EVAL=false
                shift
                ;;
            --eval-only)
                DO_SETUP=false
                DO_SPLIT=false
                DO_TRAIN=false
                shift
                ;;
            --tensorboard)
                DO_SETUP=false
                DO_SPLIT=false
                DO_TRAIN=false
                DO_EVAL=false
                run_tensorboard
                exit 0
                ;;
            --all)
                DO_SETUP=true
                DO_SPLIT=true
                DO_TRAIN=true
                DO_EVAL=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Execute pipeline
    if [ "$DO_SETUP" = true ]; then
        setup_venv
        install_dependencies
    fi
    
    if [ "$DO_SPLIT" = true ]; then
        split_dataset
    fi
    
    if [ "$DO_TRAIN" = true ]; then
        run_training
    fi
    
    if [ "$DO_EVAL" = true ]; then
        run_bertscore_evaluation
        run_llm_judge_evaluation
    fi
    
    print_header "Pipeline Complete!"
    echo ""
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Files Generated:"
    echo "  - Model: $OUTPUT_DIR/final_model"
    echo "  - BERTScore: $OUTPUT_DIR/bertscore_evaluation_results.json"
    echo "  - LLM-Judge: $OUTPUT_DIR/llm_judge_evaluation_results.json"
    echo "  - TensorBoard: $OUTPUT_DIR/tensorboard"
    echo ""
    echo "Next Steps:"
    echo "  1. View logs: tensorboard --logdir $OUTPUT_DIR/tensorboard"
    echo "  2. Check BERTScore: cat $OUTPUT_DIR/bertscore_evaluation_results.json"
    echo "  3. Check LLM-Judge: cat $OUTPUT_DIR/llm_judge_evaluation_results.json"
}

# Run main
main "$@"
