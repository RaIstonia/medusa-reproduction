#! /bin/bash

# bash ./scripts/run_ddp.sh

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

use_cuda_managed_memory=1 mpirun -np 1 python train_ddp.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_weights_path ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.jtr \
    --local_dataset_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output_dir ./medusa_checkpoints_ddp \
    --batch_size 1 
    # --seq_len 128 \
    # "$@"