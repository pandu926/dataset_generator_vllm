#!/bin/bash
# ============================================================================
# Dataset Generator Runner
# Generates 150 high-quality conversations per category
# ============================================================================

set -e

echo "=============================================="
echo "  DATASET EXPANSION: 150 per Category"
echo "=============================================="
echo ""

# Create output directory
mkdir -p data/raw/categories

# Track start time
START_TIME=$(date +%s)

# Generate for each category
echo "[1/4] Generating Alur Pendaftaran (150 scenarios)..."
python generate_alur_pendaftaran.py

echo ""
echo "[2/4] Generating Informasi Umum (150 scenarios)..."
python generate_informasi_umum.py

echo ""
echo "[3/4] Generating Out of Topic (150 scenarios)..."
python generate_oot.py

echo ""
echo "[4/4] Generating Profil UNSIQ (150 scenarios)..."
python generate_profil_unsiq.py

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "  GENERATION COMPLETE!"
echo "  Duration: ${MINUTES}m ${SECONDS}s"
echo "=============================================="
echo ""
echo "Output files:"
ls -la data/raw/categories/*.json 2>/dev/null || echo "No files generated yet"
echo ""
echo "Next steps:"
echo "  1. Run format_dataset.py to format the output"
echo "  2. Run clean_and_merge.py to merge all categories"
