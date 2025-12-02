#! /bin/bash

python train_medusa1.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_weights_path ./vicuna-jittor-weights/vicuna-7b-v1.3.jtr \
    --output_dir ./medusa_checkpoints \
    --epochs 3 \
    --batch_size 2 \
    --lr 0.001