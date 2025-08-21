#!/bin/bash

# DanceBa Setup Script for macOS Silicon (M1/M2/M3)
# This script sets up the environment for running DanceBa inference on Apple Silicon

set -e  # Exit on error

echo "======================================"
echo "DanceBa Setup for macOS Silicon"
echo "======================================"

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "⚠️  Warning: Not running on Apple Silicon (detected: $ARCH)"
    echo "This setup is optimized for Apple Silicon (M1/M2/M3) processors."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Detected Apple Silicon ($ARCH)"
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi
echo "✅ Homebrew found"

# Check for Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✅ Conda found"

# Create conda environment
ENV_NAME="danceba-silicon"
echo ""
echo "Creating conda environment: $ENV_NAME"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Activating existing environment..."
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
    fi
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "Installing dependencies..."
echo "=========================="

# Install system dependencies via Homebrew
echo "Installing system dependencies..."
brew install ffmpeg portaudio 2>/dev/null || true

# Install PyTorch for Apple Silicon
echo ""
echo "Installing PyTorch with MPS support..."
pip install --upgrade pip

# Install PyTorch (stable version with MPS support)
pip install torch torchvision torchaudio

# Verify MPS support
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())" || {
    echo "⚠️  Warning: MPS backend not available. Installing nightly build..."
    pip uninstall torch torchvision torchaudio -y
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
}

# Install audio processing libraries
echo ""
echo "Installing audio processing libraries..."
pip install librosa==0.10.1 soundfile

# Install scientific computing libraries
echo ""
echo "Installing scientific computing libraries..."
pip install numpy scipy scikit-learn

# Install computer vision libraries
echo ""
echo "Installing computer vision libraries..."
pip install opencv-python pillow

# Install other required libraries
echo ""
echo "Installing additional libraries..."
pip install timm easydict smplx tqdm matplotlib

# Install optimization libraries for Apple Silicon
echo ""
echo "Installing optimization libraries..."
pip install accelerate  # Hugging Face acceleration library

# Optional: Install Core ML tools for potential future optimization
read -p "Install Core ML tools for model conversion? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install coremltools
fi

# Create required directories
echo ""
echo "Creating project directories..."
mkdir -p data experiments outputs logs

# Download test audio if needed
if [ ! -f "test_audio.wav" ]; then
    echo ""
    echo "Downloading test audio file..."
    # You can add a test audio download here
    # curl -o test_audio.wav [URL]
fi

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
    sys.exit(1)
EOF

# Create a simple test script
cat > test_inference.py << 'EOF'
#!/usr/bin/env python
"""Simple test script for DanceBa inference on macOS Silicon"""

import torch
import numpy as np

def test_device():
    """Test device availability"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using device: {device}")
        
        # Test tensor operations
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        z = torch.matmul(x, y)
        print(f"✅ Tensor operations working on {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using device: {device} (MPS not available)")
    
    return device

def test_memory():
    """Test memory allocation"""
    device = test_device()
    
    # Try allocating different sizes
    sizes = [(100, 100), (1000, 1000), (5000, 5000)]
    for size in sizes:
        try:
            x = torch.randn(*size, device=device)
            print(f"✅ Allocated tensor of size {size}")
            del x
        except Exception as e:
            print(f"⚠️  Failed to allocate {size}: {e}")
            break

if __name__ == "__main__":
    print("Testing DanceBa setup on macOS Silicon...")
    print("=" * 50)
    test_memory()
    print("=" * 50)
    print("Setup test complete!")
EOF

chmod +x test_inference.py

# Final instructions
echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the setup:"
echo "  python test_inference.py"
echo ""
echo "To run inference:"
echo "  python inference_macos.py --model_path <path> --audio_path <path>"
echo ""
echo "For optimal performance:"
echo "  - Close unnecessary applications to free up memory"
echo "  - Use Activity Monitor to track GPU usage"
echo "  - Consider using smaller batch sizes for limited memory"
echo ""
echo "Happy dancing! 🕺💃"