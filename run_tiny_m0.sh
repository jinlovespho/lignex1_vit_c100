#!/bin/bash

DATA_ARGS="
    --dataset_path /media/data1/CIFAR100
"


LOG_ARGS="
    --project-name 20240627_vit_c100
    --exp-name test
"

# 0: Linear
# 1: Feature-wise linear
# 2: Feature-wise conv
# 3: Shuffle
# 4: Shift
# 5: Averaging

MODEL_ARGS="
    --model-name vit_splithead
    --vit-type vit_tiny
    --head-mix-method 0
"

ETC_ARGS="
    --label-smoothing 
    --autoaugment 
    --batch-size 64
    --lr 0.0005 
    --weight-decay 0.0001 
    --dropout 0.1
    --warmup-epoch 5
"

CUDA_VISIBLE_DEVICES=1 python main.py  ${DATA_ARGS} ${LOG_ARGS} ${MODEL_ARGS} ${ETC_ARGS}

