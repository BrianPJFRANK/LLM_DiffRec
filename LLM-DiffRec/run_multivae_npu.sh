#!/bin/bash

# 設定參數
DATASET="amazon-Software_coldstart"
DATA_PATH="../datasets/"
LR=0.001
BATCH_SIZE=500
EPOCHS=200
GPU_ID="0"

echo "=================================================="
echo "🚀 開始運行 Baseline A: MultiVAE"
echo "數據集: ${DATASET} | 學習率: ${LR} | Batch Size: ${BATCH_SIZE}"
echo "硬體: Huawei Ascend 910 NPU (ID: ${GPU_ID})"
echo "=================================================="

python main_multivae.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --cuda \
    --gpu ${GPU_ID}