#!/bin/bash
# =============================================================================
# Dataset Generator Setup & Run Script
# UNSIQ Multi-turn Conversation Dataset Generation
# =============================================================================
# Usage: ./run_dataset_generator.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  UNSIQ Dataset Generator Setup & Run${NC}"
echo -e "${BLUE}============================================${NC}"

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_dataset"
PYTHON_VERSION="python3"

# =============================================================================
# Step 1: Check Python
# =============================================================================
echo -e "\n${YELLOW}[1/5] Checking Python installation...${NC}"
if ! command -v ${PYTHON_VERSION} &> /dev/null; then
    echo -e "${RED}Error: Python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VER=$(${PYTHON_VERSION} --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: ${PYTHON_VER}${NC}"

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================
echo -e "\n${YELLOW}[2/5] Setting up virtual environment...${NC}"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating new venv at ${VENV_DIR}..."
    ${PYTHON_VERSION} -m venv "${VENV_DIR}"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Using existing venv at ${VENV_DIR}${NC}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
pip install --upgrade pip --quiet

# =============================================================================
# Step 3: Install Dependencies from requirements.txt
# =============================================================================
echo -e "\n${YELLOW}[3/5] Installing dependencies from requirements.txt...${NC}"

echo "Installing packages (this may take several minutes)..."
pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet

# Verify critical packages
echo -e "\n${YELLOW}Verifying installations...${NC}"
${PYTHON_VERSION} -c "import torch; print(f'  PyTorch: {torch.__version__}')"
${PYTHON_VERSION} -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
${PYTHON_VERSION} -c "import vllm; print(f'  vLLM: {vllm.__version__}')" 2>/dev/null || echo "  vLLM: (will verify at runtime)"
echo -e "${GREEN}✓ Dependencies installed${NC}"

# =============================================================================
# Step 4: Check CUDA
# =============================================================================
echo -e "\n${YELLOW}[4/5] Checking CUDA/GPU...${NC}"
${PYTHON_VERSION} << 'CUDA_CHECK'
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  ✓ CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("  ⚠ CUDA not available - will run on CPU (slow)")
CUDA_CHECK

# =============================================================================
# Step 5: Run Dataset Generator
# =============================================================================
echo -e "\n${YELLOW}[5/5] Running Dataset Generator...${NC}"
echo -e "${BLUE}============================================${NC}"
echo "  Seed: 758"
echo "  Target: 2500 samples"
echo "  Output: data/raw/synthetic_multiturn_v1.json"
echo -e "${BLUE}============================================${NC}\n"

cd "${SCRIPT_DIR}"
${PYTHON_VERSION} main.py

# =============================================================================
# Done
# =============================================================================
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  Dataset generation complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Output file: ${SCRIPT_DIR}/data/raw/synthetic_multiturn_v1.json"
echo ""

# Deactivate venv
deactivate
