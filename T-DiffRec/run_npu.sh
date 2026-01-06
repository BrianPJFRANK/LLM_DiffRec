#!/bin/bash
# run_npu_tdiffrec.sh - Training script for T-DiffRec on NPU
# Purpose: Execute the actual training job with specified parameters for T-DiffRec
# Assumes: Environment is already set up (run setup_env.sh first)
# Usage: bash run_npu_tdiffrec.sh [dataset] [lr] [wd] [batch_size] ... [w_max] [gpu_id]

# ==============================================
# SECTION 1: ENVIRONMENT SETUP FOR THIS JOB
# ==============================================

# Activate Ascend NPU environment (required for every training session)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set Python path to include current directory and subdirectories
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ==============================================
# SECTION 2: PARAMETER PARSING - T-DiffRec SPECIFIC
# ==============================================

# Extract command line arguments for T-DiffRec
# Note: T-DiffRec has w_min and w_max parameters for temporal reweighting
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
W_MIN=${14}         # Minimum weight for temporal interactions (e.g., 0.1)
W_MAX=${15}         # Maximum weight for temporal interactions (e.g., 1.0)
LOG_NAME=${16}      # Log file name identifier (e.g., "log")
ROUND=${17}         # Experiment round number (e.g., 1)
GPU=${18}           # NPU device ID (e.g., 0)

# ==============================================
# SECTION 3: PRE-RUN CHECKS AND SETUP
# ==============================================

# Create log directory for this dataset if it doesn't exist
mkdir -p ./log/$DATASET

# Generate timestamp for unique log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Display training configuration - T-DiffRec specific
echo "========================================"
echo "T-DiffRec TRAINING STARTING - NPU VERSION"
echo "========================================"
echo "Dataset:        $DATASET"
echo "Learning Rate:  $LR"
echo "Batch Size:     $BATCH_SIZE"
echo "Diffusion Steps: $STEPS"
echo "Temporal Weight Range: [$W_MIN, $W_MAX]"
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
# SECTION 4: EXECUTE T-DiffRec TRAINING
# ==============================================

echo "Starting T-DiffRec training process..."
echo "Note: Using temporal reweighting with w_min=$W_MIN, w_max=$W_MAX"

# Execute the main training script with all parameters including w_min and w_max
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
    --w_min=$W_MIN \
    --w_max=$W_MAX \
    --log_name=$LOG_NAME \
    --round=$ROUND \
    --gpu=$GPU 2>&1 | tee ./log/$DATASET/${ROUND}_${DATASET}_tdiffrec_${TIMESTAMP}_npulog.txt

# ==============================================
# SECTION 5: POST-RUN SUMMARY
# ==============================================

TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ T-DiffRec TRAINING COMPLETED SUCCESSFULLY"
    echo "========================================"
    echo "Log file saved: ./log/$DATASET/${ROUND}_${DATASET}_tdiffrec_${TIMESTAMP}_npulog.txt"
    echo "Models saved: ./saved_models/"
    echo ""
    echo "T-DiffRec specific parameters used:"
    echo "  w_min: $W_MIN (minimum temporal weight)"
    echo "  w_max: $W_MAX (maximum temporal weight)"
    echo "  reweight: $REWEIGHT"
else
    echo ""
    echo "========================================"
    echo "✗ T-DiffRec TRAINING FAILED (Exit code: $TRAINING_STATUS)"
    echo "========================================"
    echo "Check the log file for error details:"
    echo "  ./log/$DATASET/${ROUND}_${DATASET}_tdiffrec_${TIMESTAMP}_npulog.txt"
    echo ""
    echo "T-DiffRec specific troubleshooting:"
    echo "1. Check w_min < w_max (should be between 0 and 1)"
    echo "2. Verify dataset has timestamps for temporal modeling"
    echo "3. Ensure reweight parameter is set correctly (1=True, 0=False)"
fi

echo ""
echo "NPU Resource cleanup..."
# Optional: Clear NPU cache if needed
# torch_npu.npu.empty_cache()