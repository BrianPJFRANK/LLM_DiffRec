#!/bin/bash
# setup_env.sh - One-time environment setup for DiffRec on Huawei NPU
# Purpose: Install dependencies, check environment, create directories
# Run this ONCE before starting any training jobs

echo "=== Setting up DiffRec NPU Environment ==="

# 1. Activate Ascend environment (prerequisite for NPU operations)
echo "1. Activating Ascend environment..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. Install required Python packages
echo "2. Installing Python dependencies..."
pip install scipy bottleneck tqdm --quiet
if [ $? -eq 0 ]; then
    echo "   Dependencies installed successfully"
else
    echo "   Warning: Some dependencies failed to install"
fi

# 3. Verify NPU availability
echo "3. Checking NPU support..."
python -c "
import torch
import torch_npu
print(f'PyTorch Version: {torch.__version__}')
print(f'NPU Available: {torch.npu.is_available()}')
if torch.npu.is_available():
    print(f'NPU Device Count: {torch.npu.device_count()}')
    for i in range(torch.npu.device_count()):
        print(f'  Device {i}: {torch.npu.get_device_name(i)}')
else:
    print('  ERROR: NPU not available. Check your AI Studio environment configuration.')
"

# 4. Create necessary directories for the project
echo "4. Creating directory structure..."
mkdir -p ./log ./saved_models ./checkpoints
echo "   Created: ./log/ (for training logs)"
echo "   Created: ./saved_models/ (for trained models)"
echo "   Created: ./checkpoints/ (for pretrained models)"

# 5. Check dataset directory
echo "5. Checking dataset directory..."
if [ -d "./datasets" ]; then
    echo "   Found ./datasets/ directory"
    echo "   Available datasets:"
    ls ./datasets/ 2>/dev/null || echo "     (empty or inaccessible)"
else
    echo "   WARNING: ./datasets/ directory not found!"
    echo "   Please upload your datasets to ./datasets/"
fi

echo ""
echo "=== Environment Setup Complete ==="
echo "Next steps:"
echo "1. Upload your datasets to ./datasets/"
echo "2. Run your training with: bash run_npu.sh [parameters]"
echo "3. Monitor logs in ./log/"