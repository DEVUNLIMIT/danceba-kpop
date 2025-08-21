#!/bin/bash

# DanceBa Setup Script for macOS Silicon using venv
# Alternative setup using Python venv instead of conda

set -e  # Exit on error

echo "======================================"
echo "DanceBa Setup for macOS Silicon (venv)"
echo "======================================"

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "⚠️  Warning: Not running on Apple Silicon (detected: $ARCH)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Detected Apple Silicon ($ARCH)"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $PYTHON_VERSION"

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create virtual environment
VENV_NAME="venv_silicon"
echo ""
echo "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        python3 -m venv "$VENV_NAME"
    fi
else
    python3 -m venv "$VENV_NAME"
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"

echo ""
echo "Installing dependencies..."
echo "=========================="

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Apple Silicon
echo ""
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Verify MPS support
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())" || {
    echo "⚠️  Warning: MPS backend not available."
}

# Install audio processing libraries
echo ""
echo "Installing audio processing libraries..."
pip install librosa soundfile

# Install scientific computing libraries
echo ""
echo "Installing scientific computing libraries..."
pip install numpy scipy scikit-learn pyyaml

# Install computer vision libraries
echo ""
echo "Installing computer vision libraries..."
pip install opencv-python pillow

# Install other required libraries
echo ""
echo "Installing additional libraries..."
pip install timm easydict tqdm matplotlib

# Note about missing libraries
echo ""
echo "⚠️  Note: smplx requires manual installation"
echo "   Visit: https://github.com/vchoutas/smplx"

# Create required directories
echo ""
echo "Creating project directories..."
mkdir -p ../test_data outputs models logs

# Verify installation
echo ""
echo "======================================"
echo "Verifying installation..."
echo "======================================"

python << EOF
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"MPS Test: {x}")
    print("✅ MPS backend working correctly!")
else:
    print("⚠️  MPS backend not available, will use CPU")

# Test other imports
try:
    import librosa
    import cv2
    import numpy as np
    print("✅ All required libraries imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
EOF

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
# Quick activation script
source venv_silicon/bin/activate
echo "✅ Virtual environment activated: venv_silicon"
echo "   Python: $(which python)"
echo "   To deactivate: deactivate"
EOF
chmod +x activate.sh

# Final instructions
echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "To activate the environment:"
echo "  source $PROJECT_ROOT/$VENV_NAME/bin/activate"
echo "  # or use: source activate.sh"
echo ""
echo "To test MPS functionality:"
echo "  python tests/test_mps.py"
echo ""
echo "To run inference:"
echo "  python run.py ../test_data/atheart_goodgirl.mp3"
echo ""
echo "Happy dancing! 🕺💃"