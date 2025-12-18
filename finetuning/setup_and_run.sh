#!/bin/bash
# =============================================================================
# COMPLETE SETUP & RUN SCRIPT
# QLoRA SFT Fine-tuning for Gemma 3-1B-IT
# =============================================================================
# 
# This script handles everything:
# 1. Virtual environment setup
# 2. Dependencies installation
# 3. Dataset preparation
# 4. Model training
# 5. Evaluation with BERTScore
#
# Usage: ./setup_and_run.sh [OPTIONS]
#   --setup-only     Only setup environment, don't train
#   --train-only     Skip setup, only train
#   --eval-only      Skip training, only evaluate
#   --resume PATH    Resume training from checkpoint
#   --help           Show this help message
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNING_DIR="$PROJECT_ROOT/finetuning"
VENV_DIR="$PROJECT_ROOT/venv"
DATASET_PATH="$PROJECT_ROOT/data/final/multiturn_dataset_cleaned.json"
OUTPUT_DIR="$FINETUNING_DIR/outputs/gemma3-1b-qlora-sft"

# CUDA
CUDA_VERSION="12.6"
CUDA_PATH="/usr/local/cuda-$CUDA_VERSION"

# Training Configuration (can be overridden via environment variables)
MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# Evaluation Configuration
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

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

check_dataset() {
    log_info "Checking dataset..."
    if [ -f "$DATASET_PATH" ]; then
        local count=$(python3 -c "import json; print(len(json.load(open('$DATASET_PATH'))))")
        log_success "Dataset found: $DATASET_PATH ($count samples)"
    else
        log_error "Dataset not found: $DATASET_PATH"
        log_info "Please run the data generation pipeline first"
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
        log_info "Using existing venv (use --force-setup to recreate)"
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
    
    # Install latest compatible versions
    log_info "Installing HuggingFace stack (latest)..."
    pip install transformers accelerate peft trl bitsandbytes -q --upgrade
    
    log_info "Installing other dependencies..."
    pip install datasets tensorboard bert-score -q
    
    # Verify installation
    log_info "Verifying installation..."
    python3 -c "
import torch
import transformers
import accelerate
import peft
import trl
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
"
    
    log_success "All dependencies installed!"
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

run_training() {
    print_header "Step 3: Running QLoRA SFT Training"
    
    activate_venv
    check_dataset
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=0
    export TOKENIZERS_PARALLELISM=false
    
    log_info "Training Configuration:"
    echo "  Model: $MODEL_NAME"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
    echo "  Max Seq Length: $MAX_SEQ_LENGTH"
    echo "  Output: $OUTPUT_DIR"
    
    # Resume from checkpoint if specified
    RESUME_ARG=""
    if [ -n "$RESUME_CHECKPOINT" ]; then
        log_info "Resuming from checkpoint: $RESUME_CHECKPOINT"
        RESUME_ARG="--resume_checkpoint $RESUME_CHECKPOINT"
    fi
    
    # Run training
    cd "$FINETUNING_DIR"
    
    python3 train_qlora_sft.py \
        --dataset "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --lr $LEARNING_RATE \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --max_seq_length $MAX_SEQ_LENGTH \
        $RESUME_ARG
    
    log_success "Training completed!"
    log_info "Model saved to: $OUTPUT_DIR/final_model"
    log_info "TensorBoard: tensorboard --logdir $OUTPUT_DIR/tensorboard"
}

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

run_evaluation() {
    print_header "Step 4: Running BERTScore Evaluation"
    
    activate_venv
    
    # Check if fine-tuned model exists
    FINETUNED_PATH="$OUTPUT_DIR/final_model"
    if [ ! -d "$FINETUNED_PATH" ]; then
        log_error "Fine-tuned model not found at $FINETUNED_PATH"
        log_info "Please run training first"
        exit 1
    fi
    
    log_info "Evaluation Configuration:"
    echo "  Base Model: $MODEL_NAME"
    echo "  Fine-tuned: $FINETUNED_PATH"
    echo "  Test Samples: $EVAL_SAMPLES"
    echo "  Batch Size: $EVAL_BATCH_SIZE"
    
    cd "$FINETUNING_DIR"
    
    python3 evaluate_bertscore.py \
        --base_model "$MODEL_NAME" \
        --finetuned_path "$FINETUNED_PATH" \
        --test_dataset "$DATASET_PATH" \
        --output "$OUTPUT_DIR/evaluation_results.json" \
        --batch_size $EVAL_BATCH_SIZE \
        --max_samples $EVAL_SAMPLES
    
    log_success "Evaluation completed!"
    log_info "Results saved to: $OUTPUT_DIR/evaluation_results.json"
    
    # Display results
    if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
        echo ""
        echo "============================================================"
        echo "EVALUATION RESULTS"
        echo "============================================================"
        python3 -c "
import json
with open('$OUTPUT_DIR/evaluation_results.json') as f:
    r = json.load(f)
base = r['base_model']['bertscore']
ft = r['finetuned_model']['bertscore']
comp = r['comparison']
print(f\"Base Model F1:       {base['f1']:.4f}\")
print(f\"Fine-tuned F1:       {ft['f1']:.4f}\")
print(f\"Improvement:         {comp['f1_improvement']:+.4f} ({comp['f1_improvement_percent']:+.1f}%)\")
"
    fi
}

# =============================================================================
# INTERACTIVE INFERENCE
# =============================================================================

run_inference() {
    print_header "Step 5: Running Interactive Inference"
    
    activate_venv
    
    FINETUNED_PATH="$OUTPUT_DIR/final_model"
    if [ ! -d "$FINETUNED_PATH" ]; then
        log_error "Fine-tuned model not found at $FINETUNED_PATH"
        exit 1
    fi
    
    cd "$FINETUNING_DIR"
    python3 inference.py --model_path "$FINETUNED_PATH" --interactive
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
    echo "QLoRA SFT Fine-tuning Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup-only       Only setup environment (venv + dependencies)"
    echo "  --train-only       Skip setup, only run training"
    echo "  --eval-only        Skip training, only run evaluation"
    echo "  --inference        Run interactive inference"
    echo "  --tensorboard      Start TensorBoard"
    echo "  --resume PATH      Resume training from checkpoint"
    echo "  --all              Run complete pipeline (setup + train + eval)"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME         Model to fine-tune (default: google/gemma-3-1b-it)"
    echo "  EPOCHS            Number of training epochs (default: 3)"
    echo "  BATCH_SIZE        Per-device batch size (default: 8)"
    echo "  GRAD_ACCUM        Gradient accumulation steps (default: 8)"
    echo "  LEARNING_RATE     Learning rate (default: 2e-4)"
    echo "  LORA_R            LoRA rank (default: 32)"
    echo "  LORA_ALPHA        LoRA alpha (default: 64)"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Full pipeline"
    echo "  $0 --setup-only             # Only setup"
    echo "  EPOCHS=5 $0 --train-only    # Train with 5 epochs"
    echo "  $0 --resume ./outputs/gemma3-1b-qlora-sft/checkpoint-100"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    cd "$PROJECT_ROOT"
    
    print_header "QLoRA SFT Fine-tuning Pipeline"
    echo "Project Root: $PROJECT_ROOT"
    echo "Fine-tuning Dir: $FINETUNING_DIR"
    
    check_cuda
    
    # Parse arguments
    DO_SETUP=true
    DO_TRAIN=true
    DO_EVAL=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                DO_TRAIN=false
                DO_EVAL=false
                shift
                ;;
            --train-only)
                DO_SETUP=false
                DO_EVAL=false
                shift
                ;;
            --eval-only)
                DO_SETUP=false
                DO_TRAIN=false
                shift
                ;;
            --inference)
                DO_SETUP=false
                DO_TRAIN=false
                DO_EVAL=false
                run_inference
                exit 0
                ;;
            --tensorboard)
                DO_SETUP=false
                DO_TRAIN=false
                DO_EVAL=false
                run_tensorboard
                exit 0
                ;;
            --resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --all)
                DO_SETUP=true
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
    
    if [ "$DO_TRAIN" = true ]; then
        run_training
    fi
    
    if [ "$DO_EVAL" = true ]; then
        run_evaluation
    fi
    
    print_header "Pipeline Complete!"
    echo ""
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Next Steps:"
    echo "  1. View logs: tensorboard --logdir $OUTPUT_DIR/tensorboard"
    echo "  2. Test model: $0 --inference"
    echo "  3. Check results: cat $OUTPUT_DIR/evaluation_results.json"
}

# Run main
main "$@"
