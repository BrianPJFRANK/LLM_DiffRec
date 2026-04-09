#!/bin/bash
# run_semantic_npu.sh

# ==============================================
# SECTION 1: ENVIRONMENT SETUP
# ==============================================
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ==============================================
# SECTION 2: POSITIONAL PARAMETERS
# ==============================================
DATASET=$1          # 1: dataset name
LR=$2               # 2: learning rate
WD=$3               # 3: weight decay
BATCH_SIZE=$4       # 4: batch size
DIMS=$5             # 5: network dimensions
EMB_SIZE=$6         # 6: embedding size
MEAN_TYPE=$7        # 7: mean type (x0/eps)
STEPS=$8            # 8: diffusion steps
NOISE_SCALE=$9      # 9: noise scale
NOISE_MIN=${10}     # 10: noise min
NOISE_MAX=${11}     # 11: noise max
SAMPLING_STEPS=${12} # 12: sampling steps
REWEIGHT=${13}      # 13: reweight
LOG_NAME=${14}      # 14: log name
ROUND=${15}         # 15: experiment round
GPU=${16}           # 16: GPU/NPU ID

# 新增的4個語義參數
USE_SEMANTIC=${17:-1}           # 17: use semantic (1或0)
MODEL_TYPE=${18:-"semantic"}    # 18: model type
SEMANTIC_DIM=${19:-1024}        # 19: semantic dim
SEMANTIC_PROJ_DIM=${20:-256}    # 20: semantic projection dim

# ==============================================
# SECTION 3: DISPLAY CONFIGURATION
# ==============================================
echo "========================================"
echo "SEMANTIC DIFFREC - FIXED VERSION"
echo "========================================"
echo "Dataset:           $DATASET"
echo "Learning Rate:     $LR"
echo "Batch Size:        $BATCH_SIZE"
echo "Network Dims:      $DIMS"
echo "Use Semantic:      $USE_SEMANTIC"
echo "Model Type:        $MODEL_TYPE"
echo "Semantic Dim:      $SEMANTIC_DIM"
echo "Semantic Proj Dim: $SEMANTIC_PROJ_DIM"
echo "NPU Device:        $GPU"
echo "========================================"

# ==============================================
# SECTION 4: BUILD COMMAND
# ==============================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p ./log/${DATASET}_semantic

# 基礎命令
CMD="python -u main_semantic.py \
    --cuda \
    --dataset=$DATASET \
    --data_path=../datasets/ \
    --lr=$LR \
    --weight_decay=$WD \
    --batch_size=$BATCH_SIZE \
    --dims=\"$DIMS\" \
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
    --gpu=$GPU \
    --model_type=$MODEL_TYPE \
    --semantic_dim=$SEMANTIC_DIM \
    --semantic_proj_dim=$SEMANTIC_PROJ_DIM"

# 根據USE_SEMANTIC添加或省略--use_semantic參數
if [ "$USE_SEMANTIC" = "1" ]; then
    CMD="$CMD --use_semantic"
fi

echo "Executing command:"
echo "$CMD"

# ==============================================
# SECTION 5: EXECUTE
# ==============================================
eval $CMD 2>&1 | tee ./log/${DATASET}_semantic/${ROUND}_${LOG_NAME}_${TIMESTAMP}.txt

# ==============================================
# SECTION 6: POST-RUN SUMMARY
# ==============================================
TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ SEMANTIC TRAINING COMPLETED SUCCESSFULLY"
    echo "========================================"
    echo "Log file: ./log/${DATASET}_semantic/${ROUND}_${LOG_NAME}_${TIMESTAMP}.txt"
    echo "Models saved: ./saved_models_semantic/"
else
    echo ""
    echo "========================================"
    echo "✗ TRAINING FAILED (Exit code: $TRAINING_STATUS)"
    echo "========================================"
fi