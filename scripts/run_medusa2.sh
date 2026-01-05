#! /bin/bash

# bash ./scripts/run_lora.sh

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Jittor 显存优化：开启统一内存管理，允许显存不足时借用内存
export use_cuda_managed_memory=1 

# ================= 脚本配置 =================
# Python 脚本的文件名 (请确保和保存的文件名一致)

# 多卡配置：-np 2 表示使用 2 张显卡，请根据实际情况修改
# 如果是单卡，改为 -np 1
GPUS=1

# ================= 启动命令 =================
mpirun -np $GPUS python3 train_lora.py \
    \
    --base_model_path "../Medusa/vicuna-7b-v1.3" \
    --jittor_weights_path "./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl" \
    --local_dataset_path "../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
    --output_dir "./medusa_lora_checkpoints" \
    \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --epochs 2 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    \
    --medusa_heads 3 \
    --medusa_num_layers 1 \
    --medusa_decay 0.8 \
    \
    --enable_lora_training \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    \
    --logging_steps 1 \
    --save_steps 500 \
    --num_proc 16 \
    # --overwrite_cache # 如果需要重新处理数据，取消此行的注释