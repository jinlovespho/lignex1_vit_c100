#!/bin/bash

# server8
# conda activate pho

DATA_ARGS="
    --dataset imagenet
    --dataset_path /home/cvlab08/projects/data/imagenet
"


LOG_ARGS="
    --project-name 20240627_vit_c100
    --exp-name test
    --save_dir_path /home/cvlab08/projects/jinlovespho/log/lignex1_vit
"

# 0: Linear
# 1: Feature-wise linear
# 2: Feature-wise conv
# 3: Shuffle
# 4: Shift
# 5: Averaging

# imgnet image resol is resized to 224
# for patch_size 16, we need 14 patches for one side(h,w)
MODEL_ARGS="
    --model-name vit_splithead
    --vit-type vit_small
    --head-mix-method 3
    --patch 14
"

ETC_ARGS="
    --api-key True
    --label-smoothing 
    --autoaugment 
    --max-epochs 300 
    --batch-size 64
    --lr 0.001
    --weight-decay 0.05
    --dropout 0.1
    --warmup-epoch 5
    --gpu 2
"

CUDA_VISIBLE_DEVICES=2 python main.py  ${DATA_ARGS} ${LOG_ARGS} ${MODEL_ARGS} ${ETC_ARGS}

