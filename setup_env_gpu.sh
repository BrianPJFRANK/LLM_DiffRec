#!/bin/bash
# setup_env_gpu.sh - One-time environment setup for DiffRec on NVIDIA GPU
# Purpose: Install dependencies, check environment, create directories
# Run this ONCE before starting any training jobs

echo "=== Setting up DiffRec GPU Environment ==="

# 1. Check NVIDIA GPU status
echo "1. Checking NVIDIA GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo "   NVIDIA drivers are installed and functioning."
else
    echo "   Warning: nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi

# 2. Install required Python packages
echo "2. Installing Python dependencies..."
pip install cloudpickle ml-dtypes --quiet
pip install torch pandas numpy scipy bottleneck tqdm sentence-transformers transformers --quiet --root-user-action=ignore

if [ $? -eq 0 ]; then
    echo "   Dependencies installed successfully"
    python -c "import numpy;print(f'NumPy version: {numpy.__version__}')"
    python -c "import torch;print(f'PyTorch version: {torch.__version__}')"
    python -c "import pandas;print(f'Pandas version: {pandas.__version__}')"
else
    echo "   Warning: Some dependencies failed to install"
fi

# 3. Verify CUDA (GPU) availability in PyTorch
echo "3. Checking PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  ERROR: CUDA not available. Check your PyTorch installation or GPU drivers.')
"

# 4. Check dataset directory
echo "4. Checking dataset directory..."
if [ -d "../datasets" ]; then
    echo "   Found ../datasets/ directory"
    echo "   Available datasets:"
    ls ../datasets/ 2>/dev/null || echo "     (empty or inaccessible)"
else
    echo "   WARNING: ../datasets/ directory not found!"
    echo "   Please upload your datasets to ../datasets/"
fi

echo ""
echo "=== Environment Setup Complete ==="
echo "Next steps:"
echo "1. Upload your datasets to ../datasets/"
echo "2. Run your training script for GPU"
echo "3. Monitor logs in ./log/"