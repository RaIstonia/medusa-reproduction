#! /bin/bash

# bash ./scripts/run_ac.sh

# 设置 MPI 权限
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Jittor 显存优化
export use_cuda_managed_memory=1 

# 启动命令 (单卡测试，如需多卡请修改 -np)
mpirun -np 2 python3 train_ddp_fixed.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_weights_path ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --local_dataset_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output_dir ./medusa_checkpoints_1218 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 1e-3 \
    --medusa_heads 3 \
    --num_proc 16