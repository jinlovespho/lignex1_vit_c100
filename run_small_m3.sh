#!/bin/bash

# server8
# conda activate pho

DATA_ARGS="
    --dataset c100
    --dataset_path /home/cvlab08/projects/data/cifar100
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

# patch denotes the number of patches in one side(height,width). 
# img_height = patch*patch_size
MODEL_ARGS="
    --model-name vit_splithead
    --vit-type vit_small
    --head-mix-method 3
    --patch 8
"

ETC_ARGS="
    --api-key True
    --label-smoothing 
    --autoaugment 
    --max-epochs 400
    --batch-size 64
    --lr 0.0005 
    --weight-decay 0.0001 
    --dropout 0.1
    --warmup-epoch 5
"

CUDA_VISIBLE_DEVICES=0 python main.py  ${DATA_ARGS} ${LOG_ARGS} ${MODEL_ARGS} ${ETC_ARGS}

