#!/bin/bash

# bash ./scripts/run_test.sh

# export JT_SYNC=1      # 强制同步执行，报错会定位到具体 Python 行（调试时启用）
# export trace_py_var=3  # 打印变量追踪信息（会产生大量输出）

# 1. 设置显卡可见性
export CUDA_VISIBLE_DEVICES=0

# === [重要] 清空 Jittor 缓存，避免旧编译缓存导致的问题 ===
# 如果之前运行过其他版本的代码，缓存可能有问题
# 取消注释下面这行来清空缓存（首次运行会慢一些）
# rm -rf ~/.jittor/.cache/jittor/

SCRIPT_NAME="gen_model_answer.py"
BASE_MODEL_PATH="/ai4science-a100/shiyt/Medusa_final/Medusa/vicuna-7b-v1.3"
JITTOR_BASE_WEIGHTS="/ai4science-a100/shiyt/Medusa_final/repro/vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl"
# TODO:
MEDUSA_WEIGHTS="/ai4science-a100/shiyt/Medusa_final/repro/medusa_checkpoints_1218/checkpoint-best/medusa_lm_head.jtr"

# /ai4science-a100/shiyt/Medusa_final/Medusa/Medusa_zzl/llm_judge/data
QUESTION_FILE="/ai4science-a100/shiyt/Medusa_final/Medusa/Medusa_zzl/llm_judge/data/mt_bench/question.jsonl"

ANSWER_FILE="/ai4science-a100/shiyt/Medusa_final/repro/answer_files/medusa_vicuna_7b_jittor_3_heads.jsonl"

# === [重要] 清空之前的答案文件，避免追加模式导致的问题 ===
# 如果需要清空答案文件，取消注释下面这行
# > "$ANSWER_FILE"
echo "Note: Answer file is in APPEND mode. To start fresh, run: > $ANSWER_FILE"

# 3. 打印信息
echo "Running Medusa-Jittor Benchmark..."
echo "Base Config: $BASE_MODEL_PATH"
echo "Jittor Base Weights: $JITTOR_BASE_WEIGHTS"
echo "Medusa Weights: $MEDUSA_WEIGHTS"
echo "Output to: $ANSWER_FILE"

# 4. 执行命令
python "$SCRIPT_NAME" \
    --base_model_path "$BASE_MODEL_PATH" \
    --jittor_base_weights "$JITTOR_BASE_WEIGHTS" \
    --medusa_weights "$MEDUSA_WEIGHTS" \
    --question_file "$QUESTION_FILE" \
    --answer_file "$ANSWER_FILE" \
    --model_id "medusa_vicuna_7b_jittor_3_heads" \
    --medusa_num_heads 3 \
    --temperature 0.0 \
    --max_new_token 1024 \
    --posterior_threshold 0.09 \
    --posterior_alpha 0.3 \
    --medusa_choices "[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0]]"

echo "Done."