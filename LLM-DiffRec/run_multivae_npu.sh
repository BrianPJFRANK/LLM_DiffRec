#!/bin/bash

DATASET="amazon-Software_coldstart"
DATA_PATH="../datasets/"
LR=0.001
BATCH_SIZE=500
EPOCHS=200
GPU_ID="0"

echo "=================================================="
echo "Start running Baseline A: MultiVAE"
echo "Dataset: ${DATASET} | Learning Rate: ${LR} | Batch Size: ${BATCH_SIZE}"
echo "GPU/NPU/CPU: (ID: ${GPU_ID})"
echo "=================================================="

python main_multivae.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --cuda \
    --gpu ${GPU_ID}