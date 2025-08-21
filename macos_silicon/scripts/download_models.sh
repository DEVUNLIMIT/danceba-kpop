#!/bin/bash

# Download pretrained models for DanceBa

echo "======================================"
echo "Downloading DanceBa Pretrained Models"
echo "======================================"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

# Navigate to macos_silicon directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create models directory if it doesn't exist
mkdir -p models

echo ""
echo "Downloading models from HuggingFace..."
echo "This may take a while depending on your internet connection..."

# Download the models
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --local-dir ./models \
    --include "experiments/cc_motion_gpt/ckpt/*" \
    --include "experiments/sep_vqvae/ckpt/*" \
    fancongyi/danceba

echo ""
echo "✅ Models downloaded successfully to ./models/"
echo ""
echo "Directory structure:"
ls -la ./models/experiments/ 2>/dev/null || echo "Models directory created"

echo ""
echo "You can now run inference with:"
echo "  python run.py ../test_data/atheart_goodgirl.mp3"