#! /bin/bash

# bash ./scripts/run_train.sh

# python train_medusa1.py \
#     --base_model_path ../Medusa/vicuna-7b-v1.3 \
#     --jittor_weights_path ./vicuna-jittor-weights/vicuna-7b-v1.3.jtr \
#     --output_dir ./medusa_checkpoints \
#     --epochs 3 \
#     --batch_size 2 \
#     --local_dataset_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
#     --lr 0.001 \
#     --world_size 4


NUM_GPUS=4

# 获取脚本的所有其他参数 (e.g., --base_model_path ...)
ARGS="${@}"

# 使用 mpirun 启动分布式训练
# --allow-run-as-root 是在容器环境中常用的参数，如果不是 root 用户可以去掉
# CUDA_VISIBLE_DEVICES 用于确保 mpirun 控制的进程能看到正确的 GPU
mpirun -np $NUM_GPUS --allow-run-as-root \
    python train_medusa1.py \
    --batch_size 2 \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_weights_path ./vicuna-jittor-weights/vicuna-7b-v1.3.jtr \
    --local_dataset_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output_dir ./medusa_checkpoints_distributed \
    $ARGS