#!/bin/bash

# Run inference script for DanceBa on macOS Silicon
# Usage: ./run_inference.sh <audio_file> [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
CONFIG_DIR="$PROJECT_ROOT/configs"
OUTPUT_DIR="$PROJECT_ROOT/outputs"
MODEL_DIR="$PROJECT_ROOT/models"

# Default values
CONFIG_FILE="$CONFIG_DIR/config_macos_silicon.yaml"
USE_MPS=true
BENCHMARK=false

# Function to display usage
usage() {
    echo "Usage: $0 <audio_file> [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model PATH      Path to model checkpoint (default: $MODEL_DIR)"
    echo "  -o, --output PATH     Output directory (default: $OUTPUT_DIR)"
    echo "  -c, --config PATH     Config file (default: $CONFIG_FILE)"
    echo "  --cpu                 Use CPU instead of MPS"
    echo "  --benchmark           Run performance benchmark"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 music.wav"
    echo "  $0 music.wav --model ./custom_model --output ./results"
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

AUDIO_FILE="$1"
shift

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    exit 1
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --cpu)
            USE_MPS=false
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Activate conda environment
echo -e "${GREEN}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate danceba-silicon || {
    echo -e "${YELLOW}Environment not found. Running setup...${NC}"
    bash "$SCRIPT_DIR/setup_macos_silicon.sh"
    conda activate danceba-silicon
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
AUDIO_BASENAME=$(basename "$AUDIO_FILE" .wav)
OUTPUT_NAME="${AUDIO_BASENAME}_${TIMESTAMP}"

# Build Python command
PYTHON_CMD="python $SRC_DIR/inference_macos.py"
PYTHON_CMD="$PYTHON_CMD --model_path $MODEL_DIR"
PYTHON_CMD="$PYTHON_CMD --audio_path $AUDIO_FILE"
PYTHON_CMD="$PYTHON_CMD --output_path $OUTPUT_DIR/${OUTPUT_NAME}.npz"

if [ "$USE_MPS" = false ]; then
    PYTHON_CMD="$PYTHON_CMD --use_cpu"
fi

if [ "$BENCHMARK" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --benchmark"
fi

# Run inference
echo -e "${GREEN}Running inference...${NC}"
echo "Command: $PYTHON_CMD"
echo ""

# Execute with error handling
if $PYTHON_CMD; then
    echo ""
    echo -e "${GREEN}✅ Inference completed successfully!${NC}"
    echo "Output saved to: $OUTPUT_DIR/${OUTPUT_NAME}.npz"
    
    # Optional: Convert to other formats
    if command -v python &> /dev/null; then
        echo ""
        echo "Converting output formats..."
        python -c "
import numpy as np
import json

# Load output
data = np.load('$OUTPUT_DIR/${OUTPUT_NAME}.npz')

# Save as JSON (metadata only)
metadata = {
    'fps': float(data['fps']),
    'num_frames': int(data['num_frames']),
    'shape': list(data['poses'].shape)
}
with open('$OUTPUT_DIR/${OUTPUT_NAME}_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Metadata saved to: $OUTPUT_DIR/${OUTPUT_NAME}_metadata.json')
"
    fi
else
    echo ""
    echo -e "${RED}❌ Inference failed!${NC}"
    exit 1
fi