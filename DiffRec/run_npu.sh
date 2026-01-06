#!/bin/bash
# run_npu.sh - Training script for DiffRec on NPU
# Purpose: Execute the actual training job with specified parameters
# Assumes: Environment is already set up (run setup_env.sh first)
# Usage: bash run_npu.sh [dataset] [lr] [wd] [batch_size] ... [gpu_id]

# ==============================================
# SECTION 1: ENVIRONMENT SETUP FOR THIS JOB
# ==============================================

# Activate Ascend NPU environment (required for every training session)
# This sets PATH, LD_LIBRARY_PATH, and other environment variables for NPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set Python path to include current directory and subdirectories
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ==============================================
# SECTION 2: PARAMETER PARSING
# ==============================================

# Extract command line arguments
# These correspond to the hyperparameters in main.py
DATASET=$1          # Dataset name (e.g., "yelp_clean")
LR=$2               # Learning rate (e.g., 0.0001)
WD=$3               # Weight decay (e.g., 0.0)
BATCH_SIZE=$4       # Batch size (e.g., 400)
DIMS=$5             # Network dimensions as string (e.g., "[1000]")
EMB_SIZE=$6         # Embedding size (e.g., 10)
MEAN_TYPE=$7        # Mean type for diffusion (e.g., "x0")
STEPS=$8            # Diffusion steps (e.g., 5)
NOISE_SCALE=$9      # Noise scale (e.g., 0.0001)
NOISE_MIN=${10}     # Minimum noise (e.g., 0.0005)
NOISE_MAX=${11}     # Maximum noise (e.g., 0.005)
SAMPLING_STEPS=${12} # Sampling steps during inference (e.g., 0)
REWEIGHT=${13}      # Whether to use reweighting (e.g., 1 for True)
LOG_NAME=${14}      # Log file name identifier (e.g., "log")
ROUND=${15}         # Experiment round number (e.g., 1)
GPU=${16}           # NPU device ID (e.g., 0) - on NPU, usually just "0"

# ==============================================
# SECTION 3: PRE-RUN CHECKS AND SETUP
# ==============================================

# Create log directory for this dataset if it doesn't exist
mkdir -p ./log/$DATASET

# Generate timestamp for unique log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Display training configuration
echo "========================================"
echo "DIFFREC TRAINING STARTING - NPU VERSION"
echo "========================================"
echo "Dataset:        $DATASET"
echo "Learning Rate:  $LR"
echo "Batch Size:     $BATCH_SIZE"
echo "Diffusion Steps: $STEPS"
echo "NPU Device:     $GPU"
echo "Timestamp:      $TIMESTAMP"
echo "Log Directory:  ./log/$DATASET/"
echo "========================================"

# Quick check: verify NPU is accessible
python -c "
import torch
import torch_npu
if not torch.npu.is_available():
    print('ERROR: NPU not detected!')
    print('Check: 1) NPU resource allocated 2) Ascend environment activated')
    exit(1)
else:
    print(f'✓ NPU Device {torch.npu.current_device()} is ready')
"

# ==============================================
# SECTION 4: EXECUTE TRAINING
# ==============================================

echo "Starting training process..."
echo "Command: python main.py with above parameters"

# Execute the main training script with all parameters
# Note: Using tee to both display output and save to log file
python -u main.py \
    --cuda \
    --dataset=$DATASET \
    --data_path=../datasets/$DATASET/ \
    --lr=$LR \
    --weight_decay=$WD \
    --batch_size=$BATCH_SIZE \
    --dims="$DIMS" \
    --emb_size=$EMB_SIZE \
    --mean_type=$MEAN_TYPE \
    --steps=$STEPS \
    --noise_scale=$NOISE_SCALE \
    --noise_min=$NOISE_MIN \
    --noise_max=$NOISE_MAX \
    --sampling_steps=$SAMPLING_STEPS \
    --reweight=$REWEIGHT \
    --log_name=$LOG_NAME \
    --round=$ROUND \
    --gpu=$GPU 2>&1 | tee ./log/$DATASET/${ROUND}_${DATASET}_${TIMESTAMP}_npulog.txt

# ==============================================
# SECTION 5: POST-RUN SUMMARY
# ==============================================

# Check exit status of the training job
TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
    echo "========================================"
    echo "Log file saved: ./log/$DATASET/${ROUND}_${DATASET}_${TIMESTAMP}_npulog.txt"
    echo "Models saved: ./saved_models/"
    echo "Next: Run inference with trained models"
else
    echo ""
    echo "========================================"
    echo "✗ TRAINING FAILED (Exit code: $TRAINING_STATUS)"
    echo "========================================"
    echo "Check the log file for error details:"
    echo "  ./log/$DATASET/${ROUND}_${DATASET}_${TIMESTAMP}_npulog.txt"
    echo ""
    echo "Common issues:"
    echo "1. Dataset path incorrect"
    echo "2. NPU memory insufficient (try smaller batch size)"
    echo "3. Missing dependencies"
fi

echo ""
echo "NPU Resource cleanup..."
# Optional: Clear NPU cache if needed
# torch_npu.npu.empty_cache()