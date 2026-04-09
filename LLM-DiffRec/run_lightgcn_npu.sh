#!/bin/bash

DATASET="amazon-instruments"
DATA_PATH="../datasets/"
LR=0.001
BATCH_SIZE=1024
EPOCHS=300
LAYERS=3
GPU_ID="0"

echo "=================================================="
echo "Start running Baseline B: LightGCN (${LAYERS} layers)"
echo "Dataset: ${DATASET} | Learning Rate: ${LR} | Batch Size: ${BATCH_SIZE}"
echo "GPU/NPU/CPU: (ID: ${GPU_ID})"
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