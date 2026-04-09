#!/bin/bash

# 設定參數
DATASET="amazon-instruments"
DATA_PATH="../datasets/"
LR=0.001
BATCH_SIZE=1024
EPOCHS=300
LAYERS=3
GPU_ID="0"

echo "=================================================="
echo "🚀 Start running Baseline B: LightGCN (${LAYERS} 層傳播)"
echo "數據集: ${DATASET} | 學習率: ${LR} | Batch Size: ${BATCH_SIZE}"
echo "硬體: Huawei Ascend 910 NPU (ID: ${GPU_ID})"
echo "=================================================="

python main_lightgcn.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --cuda \
    --gpu ${GPU_ID}