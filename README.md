# Medusa 模型复现仓库

本仓库实现了基于 Jittor 框架的 Medusa 模型训练与测试流程。Medusa 是一种用于加速大语言模型推理的方法，通过在基础模型上附加多个预测头来并行生成多个 token，从而提升生成速度。

## Quick Start

### 环境准备

确保已安装 Jittor 框架和必要的依赖，我们使用的是python3.9，jittor环境非常难以配置，建议使用虚拟环境并保留好每个阶段的yaml：

```bash
pip install jittor transformers datasets
```

### 训练标准 Medusa1 模型

```bash
python3 train_ddp_fixed.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_weights_path ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --output_dir ./medusa_checkpoints \
    --data_path ./data/train_data.json
```

### 训练标准 Medusa2 模型

```bash
python train_lora.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --output_dir ./medusa_checkpoints \
    --data_path ./data/train_data.json
```

### 训练 RNN 版本 Medusa 模型

```bash
python train_rnn.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --output_dir ./medusa_checkpoints_rnn \
    --data_path ./data/train_data.json
```

### 测试标准 Medusa 模型

使用 ShareGPT 格式数据测试（温度采样）：

```bash
python test_medusa_benchmark.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ./medusa_checkpoints/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096 \
    --temperature 0.7
```

使用贪婪解码测试：

```bash
python test_medusa_benchmark.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ./medusa_checkpoints/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096 \
    --greedy
```

使用 MT-bench 数据集测试：

```bash
python test_medusa_benchmark.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ./medusa_checkpoints/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../llm_judge/data/mt_bench/question.jsonl \
    --data_format mt_bench \
    --num_samples 5 \
    --max_new_token 2048
```

### 测试 RNN 版本 Medusa 模型

```bash
python test_medusa_rnn_benchmark.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ./medusa_checkpoints_rnn/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096
```

### 测试基础模型（对比基准）

```bash
python test_base_model_benchmark.py \
    --base_model_path ../vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --data_path ../ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096
```

### 查看实验结果

实验结果保存在 `results/` 目录下的 JSON 文件中，可以使用以下命令查看：

```bash
# 查看结果文件内容
cat results/benchmark_results_1.json | python -m json.tool | less
```

或者直接打开 JSON 文件查看详细的生成结果和性能指标。


## 项目结构

### 核心模型实现

- `medusa/model/` - 标准 Medusa 模型实现
- `medusa/model_rnn/` - 基于 Gated RNN 的 Medusa 变体
- `medusa/model_rnn_enhance/` - 增强版 RNN Medusa 实现(基于LSTM EAGLE等模型进行了调整改进)
- `medusa/model_moe/` - 基于 MoE（Mixture of Experts）的 Medusa 变体(某些基于rnn的尝试 效果待调整)
- `medusa/lora/` - LoRA 相关实现

### 训练脚本

- `train_lora.py` - 使用 LoRA 微调方式训练 Medusa 模型
- `train_rnn.py` - 训练 Gated RNN 版本的 Medusa 模型
- `train_rnn_enhance.py` - 训练增强版 RNN Medusa 模型
- `train_moe.py` - 训练 MoE 版本的 Medusa 模型
- `train_ddp.py` / `train_ddp_fixed.py` - 分布式训练脚本

### 测试脚本

- `test_medusa_benchmark.py` - 标准 Medusa 模型的基准测试
- `test_medusa_rnn_benchmark.py` - RNN 版本 Medusa 模型的基准测试
- `test_medusa_moe_benchmark.py` - MoE 版本 Medusa 模型的基准测试
- `test_base_model_benchmark.py` - 基础模型（无 Medusa）的基准测试
- `gen_model_answer.py` - 生成模型答案的工具脚本

### 工具脚本

- `weight_convert.py` - 权重格式转换工具
- `t_a.py` - 辅助测试
- `test_stage*.py` - 实现过程中单元测试测试脚本

### 实验结果

实验结果保存在 `results/` 目录下：

- `benchmark_results_1.json` - 标准 Medusa 模型测试结果
- `benchmark_results_2.json` - 另一组测试结果
- `benchmark_results_rnn.json` - RNN 版本模型测试结果

每个结果文件包含：
- 配置信息（模型路径、超参数等）
- Medusa 模型和基础模型的生成结果对比
- 性能指标（生成时间、token 数量等）

## 实现说明

### 架构设计

1. **模型接口设计**：移除了原先使用继承的方式处理 Mistral || Llama，转而使用在 Medusa 类中增加一个传入基础模型的接口
2. **权重管理**：传入权重时通过 medusa 和 base model 分开传输的方法
3. **精度处理**：base model 调用时使用 float16，medusa head 使用 float32，在执行时需要转换精度

### 注意力机制

1. **训练阶段**：输入一串序列，对于每个位置都进行一次预测，因此需要使用因果注意力。对于序列中存在的 pad，需要使用用户自定义的注意力（expand mask）在矩阵的最后两个维度中的（行，也就是每个查询的权重上进行广播）
2. **测试阶段**：存在 prefill 阶段，一般启用 KV Cache，也就是需要使用一个非正方的因果矩阵，需要使用根据 kv 的长度进行填充（past key values length 的内容），每次只进行一个值的输出


## 注意事项

1. **GPU 可见性处理**：代码中使用了特殊的处理方式来解决 PyTorch 和 Jittor 的库冲突，在导入 torch 时临时隐藏 GPU，导入完成后恢复
2. **精度转换**：base model 使用 float16，medusa head 使用 float32，在执行时需要确保精度转换正确
3. **权重路径**：确保 base model 权重和 medusa 权重路径正确，权重文件格式为 `.pkl`（Jittor base model）和 `.jtr`（Medusa head）


