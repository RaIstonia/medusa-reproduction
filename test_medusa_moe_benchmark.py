#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medusa 模型基准测试脚本
测试 Medusa 和 Base Model 在相同数据上的性能对比

使用ShareGPT格式数据测试：
python test_medusa_moe_benchmark.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ../repro/medusa_checkpoints_1218/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096

使用MT-bench数据集测试：
python test_medusa_moe_benchmark.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ../repro/medusa_checkpoints_moe_1230/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../Medusa/Medusa_zzl/llm_judge/data/mt_bench/question.jsonl \
    --data_format mt_bench \
    --num_samples 10 \
    --max_new_token 2048
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# --- [Hack] 解决 PyTorch 和 Jittor 的库冲突 ---
_original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import torch
from transformers import AutoTokenizer
# 恢复 GPU 可见性
if _original_cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _original_cuda_visible_devices
else:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

# --- Jittor 导入 ---
import jittor as jt
jt.flags.use_cuda = 1

# 导入模型定义
from medusa.model_moe.modeling_medusa import MedusaModel, MedusaConfig
from medusa.model_moe.modeling_llama import LlamaForCausalLM, LlamaConfig


class SimpleConversation:
    """
    简化的 Vicuna 对话模板管理器 (替代 FastChat)
    """
    def __init__(self):
        self.roles = ("USER", "ASSISTANT")
        self.sep = " "
        self.sep2 = "</s>"
        self.system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.messages = []

    def append_message(self, role, content):
        if content is None:
            self.messages.append({"role": role, "content": ""})
        else:
            self.messages.append({"role": role, "content": content})

    def get_prompt(self):
        ret = self.system + self.sep
        for msg in self.messages:
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg["content"]
            else:
                role, content = msg
            
            if role == self.roles[0]:
                ret += role + ": " + content + self.sep
            else:
                ret += role + ": " + content + self.sep2
        return ret


def load_model_and_tokenizer(base_model_path, jittor_base_weights, medusa_weights_path, medusa_num_heads=5, max_seq_length=4096, moe_num_experts=4, moe_top_k=1):
    """
    加载backbone模型、medusa heads和tokenizer
    
    Args:
        base_model_path: HuggingFace模型路径
        jittor_base_weights: Jittor格式的backbone模型权重路径
        medusa_weights_path: Medusa heads权重路径
        medusa_num_heads: Medusa heads数量
        max_seq_length: 最大序列长度（默认4096，可选4096或8192）
        moe_num_experts: MoE专家数量（必须与训练时一致）
        moe_top_k: MoE Top-K（必须与训练时一致）
    """
    print("=" * 60)
    print("正在加载模型...")
    print("=" * 60)
    
    # 1. 加载模型配置
    print(f"1. 加载配置从: {base_model_path}")
    with open(os.path.join(base_model_path, "config.json"), 'r') as f:
        config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(config_dict)
    
    # 修改最大序列长度
    original_max_len = llama_config.max_position_embeddings
    llama_config.max_position_embeddings = max_seq_length
    print(f"   修改最大序列长度: {original_max_len} -> {max_seq_length}")
    
    # 2. 加载 Tokenizer
    print("2. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    # 更新tokenizer的model_max_length
    tokenizer.model_max_length = max_seq_length
    
    # 3. 加载基座模型
    print("3. 加载 Base Model...")
    base_model = LlamaForCausalLM(llama_config)
    if jittor_base_weights and os.path.exists(jittor_base_weights):
        print(f"   从 {jittor_base_weights} 加载权重...")
        base_weights = jt.load(jittor_base_weights)
        base_model.load_parameters(base_weights)
    base_model.float16()
    base_model.eval()
    
    # 4. 加载 Medusa
    print("4. 加载 Medusa Model...")
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=1,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size,
        base_model_name_or_path=base_model_path,
        # MoE 参数
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k
    )
    model = MedusaModel(medusa_config, base_model=base_model)
    
    if medusa_weights_path and os.path.exists(medusa_weights_path):
        print(f"   从 {medusa_weights_path} 加载 Medusa heads 权重...")
        medusa_weights = jt.load(medusa_weights_path)
        # [关键修改] medusa_head -> medusa_blocks
        try:
            model.medusa_blocks.load_parameters(medusa_weights)
        except Exception as e:
            print(f"权重加载警告: {e}")
            print("尝试使用 permissive loading...")
            model.medusa_blocks.load_parameters(medusa_weights, strict=False)
    
    model.eval()
    
    print("模型加载完成！")
    print("=" * 60)
    
    return model, base_model, tokenizer


def generate_with_medusa(model, tokenizer, formatted_prompt, 
                        temperature=0.7,
                        max_new_token=512,
                        posterior_threshold=0.01,
                        posterior_alpha=0.1,
                        medusa_choices=None):
    """
    使用 Medusa 进行生成，返回生成结果、接受长度列表和生成时间
    
    Args:
        formatted_prompt: 已经格式化好的prompt（按照训练时的格式）
    """
    # 编码输入（不添加special tokens，与训练时保持一致）
    input_ids = tokenizer([formatted_prompt], return_tensors="np", add_special_tokens=False)["input_ids"]
    input_ids = jt.array(input_ids)
    input_len = input_ids.shape[1]
    
    # 记录时间
    start_time = time.time()
    
    with jt.no_grad():
        generated_ids = []
        accept_lengths = []
        step_count = 0
        
        generator = model.medusa_generate(
            input_ids,
            temperature=temperature,
            max_steps=max_new_token,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            medusa_choices=medusa_choices,
            sampling='typical',
            fast=True,
            tokenizer=tokenizer,
            debug_callback=None
        )
        
        for output in generator:
            step_count += 1
            if "ids" in output:
                generated_ids = output["ids"]
            if "accept_length" in output:
                # accept_length是medusa预测的token中被接受的数量
                # 每个step实际接受的token数 = accept_length + 1 (base model的1个token)
                accept_lengths.append(output["accept_length"] + 1)
            
            # 保存最后一次输出的性能统计信息（累计值）
            if step_count == 1 or "total_medusa_heads_time" in output:
                perf_stats = {
                    "prefill_time": output.get("prefill_time", 0.0),
                    "total_medusa_heads_time": output.get("total_medusa_heads_time", 0.0),
                    "total_tree_decoding_time": output.get("total_tree_decoding_time", 0.0),
                    "total_generate_candidates_time": output.get("total_generate_candidates_time", 0.0),
                    "total_evaluate_posterior_time": output.get("total_evaluate_posterior_time", 0.0),
                    "total_update_inputs_time": output.get("total_update_inputs_time", 0.0),
                    "total_step_overhead_time": output.get("total_step_overhead_time", 0.0),
                }
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # 计算生成的token数量
    num_generated_tokens = len(generated_ids) if generated_ids else 0
    
    # 计算吞吐量 (tokens/s)
    throughput = num_generated_tokens / generation_time if generation_time > 0 else 0.0
    
    # 计算平均每个step接受的token数量
    avg_accept_per_step = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0
    
    # 计算总步数和KV cache相关统计
    # 注意：Medusa内部管理KV cache，每个step只有一次forward：
    # Tree Decoding Forward（验证候选路径，同时计算所有候选token的hidden states）
    # 不再需要 Correction Step Forward，因为最后一个被接受token的hidden state已经在tree decoding中计算好了
    total_steps = step_count
    # 每次step有1次forward调用（Tree Decoding），加上1次prefill forward
    total_forward_calls = step_count + 1  # 每个step: Tree Decoding，+1 是 prefill
    # Medusa 总是使用 KV cache（通过 gather_and_reset 复用），所以所有 forward 都使用了 KV cache
    kv_cache_used_count = total_forward_calls
    fallback_count = 0  # Medusa 不使用 fallback
    
    # 计算性能统计（使用最后一次输出的累计值）
    perf_stats = perf_stats if 'perf_stats' in locals() else {
        "prefill_time": 0.0,
        "total_medusa_heads_time": 0.0,
        "total_tree_decoding_time": 0.0,
        "total_generate_candidates_time": 0.0,
        "total_evaluate_posterior_time": 0.0,
        "total_update_inputs_time": 0.0,
        "total_step_overhead_time": 0.0,
    }
    
    # 计算所有已统计的时间
    measured_time = (
        perf_stats.get("prefill_time", 0.0) +
        perf_stats.get("total_medusa_heads_time", 0.0) +
        perf_stats.get("total_tree_decoding_time", 0.0) +
        perf_stats.get("total_generate_candidates_time", 0.0) +
        perf_stats.get("total_evaluate_posterior_time", 0.0) +
        perf_stats.get("total_update_inputs_time", 0.0) +
        perf_stats.get("total_step_overhead_time", 0.0)
    )
    
    # 计算剩余时间（包括循环外的开销、初始化等）
    remaining_time = generation_time - measured_time
    remaining_time = max(0.0, remaining_time)  # 确保不为负
    perf_stats["remaining_time"] = remaining_time
    
    return {
        "generated_ids": generated_ids,
        "num_tokens": num_generated_tokens,
        "generation_time": generation_time,
        "throughput": throughput,
        "accept_lengths": accept_lengths,
        "avg_accept_per_step": avg_accept_per_step,
        "num_steps": step_count,
        "total_steps": total_steps,
        "total_forward_calls": total_forward_calls,
        "kv_cache_used_count": kv_cache_used_count,
        "fallback_count": fallback_count,
        # 性能统计信息
        "perf_stats": perf_stats
    }


def _validate_past_key_values(past_key_values):
    """
    验证past_key_values的结构是否正确
    """
    if past_key_values is None:
        return False
    try:
        # 检查是否是元组或列表
        if not isinstance(past_key_values, (tuple, list)):
            return False
        
        # 检查每一层是否有K和V
        for layer_past in past_key_values:
            if not isinstance(layer_past, (tuple, list)) or len(layer_past) < 2:
                return False
            k_tensor, v_tensor = layer_past[0], layer_past[1]
            
            # 检查是否是Jittor Var
            if not isinstance(k_tensor, jt.Var) or not isinstance(v_tensor, jt.Var):
                return False
            
            # 检查shape是否正确（应该有3个维度）
            if len(k_tensor.shape) < 3 or len(v_tensor.shape) < 3:
                return False
            
        return True
    except (TypeError, IndexError, AttributeError):
        return False


def generate_with_base_model(base_model, tokenizer, formatted_prompt,
                             temperature=0.7,
                             max_new_token=512):
    """
    使用 Base Model 进行生成，返回生成结果和生成时间
    
    Args:
        formatted_prompt: 已经格式化好的prompt（按照训练时的格式）
    """
    # 编码输入（不添加special tokens，与训练时保持一致）
    input_ids = tokenizer([formatted_prompt], return_tensors="np", add_special_tokens=False)["input_ids"]
    input_ids = jt.array(input_ids)
    input_len = input_ids.shape[1]
    
    # 记录时间
    start_time = time.time()
    
    # 初始化性能统计变量
    prefill_time = 0.0  # prefill阶段的时间
    decoding_forward_time = 0.0  # decoding阶段forward的总时间
    sampling_time = 0.0  # 采样token的总时间
    other_time = 0.0  # 其他操作的时间
    
    with jt.no_grad():
        generated_ids = []
        past_key_values = None
        accumulated_input_ids = input_ids  # 用于fallback方案
        kv_cache_used_count = 0  # 统计使用KV cache的次数
        fallback_count = 0  # 统计使用fallback的次数
        
        # 第一次forward：传入完整的input_ids（prefill阶段）
        prefill_start = time.time()
        outputs = base_model(
            input_ids=input_ids,
            past_key_values=None,  # 第一次调用时明确传入None
            use_cache=True,
            return_dict=True
        )
        jt.sync_all()
        prefill_time = time.time() - prefill_start
        
        # 处理返回值（可能是tuple或对象）
        if isinstance(outputs, tuple):
            logits = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
        else:
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        
        # 检查logits是否有效
        if logits is None:
            raise ValueError("Model returned None logits")
        
        # 验证past_key_values的结构
        kv_cache_valid = _validate_past_key_values(past_key_values)
        if not kv_cache_valid:
            past_key_values = None
        
        # 获取最后一个位置的logits
        next_token_logits = logits[0, -1, :]
        
        # 采样第一个token
        sampling_start = time.time()
        if temperature == 0:
            next_token_id = jt.argmax(next_token_logits, dim=-1).item()
        else:
            # 使用温度采样
            probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
            next_token_idx = jt.multinomial(probs, 1)
            next_token_id = next_token_idx.item()
        sampling_time += time.time() - sampling_start
        
        generated_ids.append(next_token_id)
        
        # 检查是否遇到EOS token
        if next_token_id == tokenizer.eos_token_id:
            end_time = time.time()
            generation_time = end_time - start_time
            num_generated_tokens = len(generated_ids)
            throughput = num_generated_tokens / generation_time if generation_time > 0 else 0.0
            other_time = generation_time - (prefill_time + decoding_forward_time + sampling_time)
            return {
                "generated_ids": generated_ids,
                "num_tokens": num_generated_tokens,
                "generation_time": generation_time,
                "throughput": throughput,
                "kv_cache_used_count": 0,
                "fallback_count": 0,
                "total_steps": 0,
                "perf_stats": {
                    "prefill_time": prefill_time,
                    "decoding_forward_time": decoding_forward_time,
                    "sampling_time": sampling_time,
                    "other_time": other_time
                }
            }
        
        # 后续生成：只传入新生成的token
        for step in range(1, max_new_token):
            # 准备下一个token的input_ids（只有新生成的token）
            next_token = jt.array([[next_token_id]], dtype="int64")
            
            # 如果past_key_values无效，使用累积的input_ids（fallback方案）
            kv_cache_valid = _validate_past_key_values(past_key_values)
            forward_start = time.time()
            if not kv_cache_valid:
                # 如果past_key_values无效，累积所有生成的token并重新forward
                fallback_count += 1
                accumulated_input_ids = jt.concat([accumulated_input_ids, next_token], dim=1)
                outputs = base_model(
                    input_ids=accumulated_input_ids,
                    past_key_values=None,
                    use_cache=True,
                    return_dict=True
                )
            else:
                # Forward pass（只传入新token，使用past_key_values）
                kv_cache_used_count += 1
                outputs = base_model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            jt.sync_all()
            decoding_forward_time += time.time() - forward_start
            
            # 处理返回值
            if isinstance(outputs, tuple):
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            
            # 检查logits是否有效
            if logits is None:
                raise ValueError(f"Model returned None logits at step {step}")
            
            # 验证past_key_values的结构
            kv_cache_valid = _validate_past_key_values(past_key_values)
            if not kv_cache_valid:
                past_key_values = None
            
            # 获取logits（只有一个位置）
            next_token_logits = logits[0, -1, :]
            
            # 采样下一个token
            sampling_start = time.time()
            if temperature == 0:
                next_token_id = jt.argmax(next_token_logits, dim=-1).item()
            else:
                # 使用温度采样
                probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
                next_token_idx = jt.multinomial(probs, 1)
                next_token_id = next_token_idx.item()
            sampling_time += time.time() - sampling_start
            
            generated_ids.append(next_token_id)
            
            # 检查是否遇到EOS token
            if next_token_id == tokenizer.eos_token_id:
                break
        
        # 计算总步数（减去第一次forward）
        total_steps = len(generated_ids) - 1
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # 计算生成的token数量
    num_generated_tokens = len(generated_ids)
    
    # 计算吞吐量 (tokens/s)
    throughput = num_generated_tokens / generation_time if generation_time > 0 else 0.0
    
    # 计算其他时间（包括循环开销、numpy转换等）
    other_time = generation_time - (prefill_time + decoding_forward_time + sampling_time)
    other_time = max(0.0, other_time)  # 确保不为负
    
    return {
        "generated_ids": generated_ids,
        "num_tokens": num_generated_tokens,
        "generation_time": generation_time,
        "throughput": throughput,
        "kv_cache_used_count": kv_cache_used_count,
        "fallback_count": fallback_count,
        "total_steps": total_steps,
        "perf_stats": {
            "prefill_time": prefill_time,
            "decoding_forward_time": decoding_forward_time,
            "sampling_time": sampling_time,
            "other_time": other_time
        }
    }


def format_conversation_for_inference(conversations, tokenizer, max_length=4096):
    """
    按照训练时的格式格式化对话，用于推理
    返回格式化后的prompt（包含所有历史对话，但不包含最后一个assistant的回复内容）
    
    格式与训练时一致：
    - system_prompt + separator
    - 对于每个turn:
      - human: "USER: " + content + separator
      - gpt (历史): "ASSISTANT: " + content + eos_token
      - gpt (最后一个): "ASSISTANT: " (只添加前缀，内容待生成)
    
    Args:
        conversations: 对话列表
        tokenizer: tokenizer实例
        max_length: 最大序列长度（默认4096，可选4096或8192）
    
    Returns:
        格式化后的prompt字符串
    """
    separator = " "
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    # 找到最后一个assistant回复的位置
    last_gpt_idx = -1
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("from") == "gpt":
            last_gpt_idx = i
            break
    
    # 构建完整的对话历史
    formatted_parts = [system_prompt + separator]
    
    # 处理所有turn，但最后一个assistant只添加前缀
    for i, turn in enumerate(conversations):
        role = turn.get("from", "")
        content = turn.get("value", "")
        
        if role == "human":
            role_str = "USER: "
            formatted_parts.append(role_str + content + separator)
        elif role == "gpt":
            role_str = "ASSISTANT: "
            if i == last_gpt_idx:
                # 最后一个assistant回复，只添加前缀（内容待生成）
                formatted_parts.append(role_str)
            else:
                # 历史assistant回复，添加完整内容 + eos_token（与训练时一致）
                formatted_parts.append(role_str + content + eos_token)
    
    # 组合完整的prompt
    full_prompt = "".join(formatted_parts)
    
    # 检查token长度，如果超过限制则截断
    input_ids = tokenizer([full_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
    
    if len(input_ids) > max_length:
        # 需要截断：策略是保留system prompt和最后一个human消息，然后添加ASSISTANT:前缀
        # 找到最后一个human消息
        last_human_idx = -1
        for i in range(len(conversations) - 1, -1, -1):
            if conversations[i].get("from") == "human":
                last_human_idx = i
                break
        
        if last_human_idx >= 0:
            # 构建最小prompt：system + 最后一个human + ASSISTANT:
            last_human_content = conversations[last_human_idx].get("value", "")
            min_prompt = system_prompt + separator + "USER: " + last_human_content + separator + "ASSISTANT: "
            min_ids = tokenizer([min_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
            
            if len(min_ids) > max_length:
                # 连最后一个human消息都太长，需要截断human内容
                # 保留system + "USER: " + 部分human内容 + " ASSISTANT: "
                system_ids = tokenizer([system_prompt + separator + "USER: "], return_tensors="np", add_special_tokens=False)["input_ids"][0]
                assistant_ids = tokenizer([" ASSISTANT: "], return_tensors="np", add_special_tokens=False)["input_ids"][0]
                available_length = max_length - len(system_ids) - len(assistant_ids)
                
                if available_length > 0:
                    human_ids = tokenizer([last_human_content], return_tensors="np", add_special_tokens=False)["input_ids"][0]
                    if len(human_ids) > available_length:
                        human_ids = human_ids[:available_length]
                        last_human_content = tokenizer.decode(human_ids, skip_special_tokens=False)
                    
                    full_prompt = system_prompt + separator + "USER: " + last_human_content + separator + "ASSISTANT: "
                else:
                    # 如果连基本结构都放不下，只保留system和ASSISTANT:
                    full_prompt = system_prompt + separator + "ASSISTANT: "
            else:
                # 可以容纳最后一个human消息，尝试添加更多历史对话
                remaining_length = max_length - len(min_ids)
                history_parts = []
                
                # 从最后一个human之前开始，往前添加历史对话
                for i in range(last_human_idx - 1, -1, -1):
                    turn = conversations[i]
                    role = turn.get("from", "")
                    content = turn.get("value", "")
                    
                    if role == "human":
                        part = "USER: " + content + separator
                    elif role == "gpt":
                        part = "ASSISTANT: " + content + eos_token
                    else:
                        continue
                    
                    # 检查添加这部分后是否超过限制
                    test_parts = [system_prompt + separator] + history_parts + [part] + ["USER: " + last_human_content + separator + "ASSISTANT: "]
                    test_prompt = "".join(test_parts)
                    test_ids = tokenizer([test_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
                    
                    if len(test_ids) <= max_length:
                        history_parts.append(part)
                    else:
                        break
                
                full_prompt = system_prompt + separator + "".join(history_parts) + "USER: " + last_human_content + separator + "ASSISTANT: "
        else:
            # 没有human消息，只保留system和ASSISTANT:
            full_prompt = system_prompt + separator + "ASSISTANT: "
        
        # 最终验证长度
        final_ids = tokenizer([full_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
        if len(final_ids) > max_length:
            # 如果还是太长，强制截断到max_length
            truncated_ids = final_ids[:max_length]
            full_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            # 确保以ASSISTANT:结尾
            if not full_prompt.rstrip().endswith("ASSISTANT:"):
                last_assistant_pos = full_prompt.rfind("ASSISTANT:")
                if last_assistant_pos >= 0:
                    full_prompt = full_prompt[:last_assistant_pos + len("ASSISTANT:")]
                else:
                    # 如果没有ASSISTANT:，添加一个
                    full_prompt = system_prompt + separator + "ASSISTANT: "
    
    return full_prompt


def convert_mt_bench_to_conversations(mt_bench_item):
    """
    将MT-bench格式转换为ShareGPT格式（conversations格式）
    
    MT-bench格式:
    {
        "question_id": 81,
        "category": "writing",
        "turns": ["question1", "question2"]
    }
    
    转换为:
    [
        {"from": "human", "value": "question1"},
        {"from": "gpt", "value": ""},  # 第一轮回答（待生成）
        {"from": "human", "value": "question2"},
        {"from": "gpt", "value": ""}   # 第二轮回答（待生成）
    ]
    """
    conversations = []
    turns = mt_bench_item.get('turns', [])
    
    for turn in turns:
        conversations.append({"from": "human", "value": turn})
        # 添加assistant的占位符（内容待生成）
        conversations.append({"from": "gpt", "value": ""})
    
    return conversations


def load_test_data(data_path, num_samples=200, tokenizer=None, data_format='auto'):
    """
    加载测试数据（前num_samples条）
    按照训练时的格式处理数据
    
    Args:
        data_path: 数据文件路径
        num_samples: 加载的样本数量
        tokenizer: tokenizer实例
        data_format: 数据格式，'auto'（自动检测）、'sharegpt'或'mt_bench'
    """
    print(f"正在加载测试数据: {data_path}")
    
    # 自动检测数据格式
    if data_format == 'auto':
        # 尝试读取第一行来判断格式
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    first_item = json.loads(first_line)
                    if 'turns' in first_item and 'question_id' in first_item:
                        data_format = 'mt_bench'
                        print("检测到MT-bench格式")
                    elif 'conversations' in first_item:
                        data_format = 'sharegpt'
                        print("检测到ShareGPT格式")
                    else:
                        # 尝试读取完整JSON文件
                        f.seek(0)
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            if 'conversations' in data[0]:
                                data_format = 'sharegpt'
                                print("检测到ShareGPT格式（JSON）")
                            else:
                                data_format = 'sharegpt'  # 默认假设是ShareGPT格式
                                print("假设为ShareGPT格式")
                        else:
                            data_format = 'sharegpt'
                            print("假设为ShareGPT格式")
        except Exception as e:
            print(f"自动检测格式失败: {e}，假设为ShareGPT格式")
            data_format = 'sharegpt'
    
    test_data = []
    
    # 根据格式加载数据
    if data_format == 'mt_bench':
        # MT-bench是JSONL格式
        print("按MT-bench格式（JSONL）读取...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    test_data.append(item)
                except json.JSONDecodeError:
                    continue
    else:
        # ShareGPT格式
        # 尝试读取JSON文件（可能是列表格式）
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果是列表，直接取前num_samples条
            if isinstance(data, list):
                test_data = data[:num_samples]
            else:
                # 如果是字典，尝试找到数据列表
                test_data = list(data.values())[:num_samples] if isinstance(data, dict) else [data][:num_samples]
        except (json.JSONDecodeError, MemoryError):
            # 如果文件太大，尝试逐行读取（JSONL格式）
            print("尝试按JSONL格式读取...")
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        test_data.append(item)
                    except json.JSONDecodeError:
                        continue
    
    print(f"加载了 {len(test_data)} 条测试数据")
    
    # 按照训练时的格式处理数据
    test_inputs = []
    for item in test_data:
        if data_format == 'mt_bench':
            # MT-bench格式：每个问题有两个turns
            # 我们为第一个turn生成测试用例（只包含第一个问题）
            # 如果需要测试第二个turn，可以在生成第一个回答后再处理
            turns = item.get('turns', [])
            if len(turns) > 0:
                # 第一个turn：只包含第一个问题
                first_conversations = [
                    {"from": "human", "value": turns[0]},
                    {"from": "gpt", "value": ""}  # 待生成
                ]
                max_length = getattr(tokenizer, 'model_max_length', 4096) if tokenizer else 4096
                formatted_prompt = format_conversation_for_inference(first_conversations, tokenizer, max_length=max_length)
                test_inputs.append({
                    'prompt': formatted_prompt,
                    'conversations': first_conversations,
                    'question_id': item.get('question_id'),
                    'category': item.get('category'),
                    'turn': 1,
                    'original_item': item  # 保留原始数据用于参考
                })
        else:
            # ShareGPT格式
            conversations = item.get('conversations', [])
            if conversations and len(conversations) > 0:
                # 确保至少有一条human消息
                has_human = any(turn.get('from') == 'human' for turn in conversations)
                if has_human:
                    # 按照训练时的格式格式化对话
                    # 使用模型的最大序列长度（从tokenizer获取，默认4096）
                    max_length = getattr(tokenizer, 'model_max_length', 4096) if tokenizer else 4096
                    formatted_prompt = format_conversation_for_inference(conversations, tokenizer, max_length=max_length)
                    test_inputs.append({
                        'prompt': formatted_prompt,
                        'conversations': conversations  # 保留原始对话用于参考
                    })
    
    print(f"提取了 {len(test_inputs)} 条测试输入")
    return test_inputs


def main():
    parser = argparse.ArgumentParser(description="Medusa 模型基准测试")
    
    # 路径参数
    parser.add_argument("--base_model_path", type=str, 
                       default="../Medusa/vicuna-7b-v1.3",
                       help="HuggingFace模型路径")
    parser.add_argument("--jittor_base_weights", type=str,
                       default="./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl",
                       help="Jittor格式的backbone模型权重")
    parser.add_argument("--medusa_weights", type=str,
                       default="./medusa_checkpoints_1218/checkpoint-best/medusa_lm_head.jtr",
                       help="Medusa heads权重路径")
    parser.add_argument("--data_path", type=str,
                       default="../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
                       help="测试数据路径")
    
    # 模型参数
    parser.add_argument("--medusa_num_heads", type=int, default=5,
                       help="Medusa heads数量")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                       choices=[4096, 8192],
                       help="最大序列长度（4096或8192，默认4096）")
    # MoE 参数
    parser.add_argument("--moe_num_experts", type=int, default=4,
                       help="MoE专家数量 (必须与训练时一致)")
    parser.add_argument("--moe_top_k", type=int, default=1,
                       help="MoE Top-K (必须与训练时一致)")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    parser.add_argument("--max_new_token", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--posterior_threshold", type=float, default=0.01,
                       help="Typical Acceptance的epsilon阈值")
    parser.add_argument("--posterior_alpha", type=float, default=0.1,
                       help="Typical Acceptance的delta参数")
    
    # 测试参数
    parser.add_argument("--num_samples", type=int, default=200,
                       help="测试样本数量")
    parser.add_argument("--data_format", type=str, default="auto",
                       choices=["auto", "sharegpt", "mt_bench"],
                       help="数据格式：auto（自动检测）、sharegpt或mt_bench")
    
    args = parser.parse_args()
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 加载模型
    model, base_model, tokenizer = load_model_and_tokenizer(
        args.base_model_path,
        args.jittor_base_weights,
        args.medusa_weights,
        args.medusa_num_heads,
        max_seq_length=args.max_seq_length,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k
    )
    
    # Warmup
    print("\n正在预热模型...")
    try:
        dummy_input = jt.array(tokenizer(["Hello"], return_tensors="np")["input_ids"])
        with jt.no_grad():
            for _ in model.medusa_generate(dummy_input, max_steps=3, tokenizer=tokenizer):
                pass
        jt.sync_all()
        jt.gc()
        print("预热完成！\n")
    except Exception as e:
        print(f"预热失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 加载测试数据（需要tokenizer来格式化）
    test_inputs = load_test_data(args.data_path, args.num_samples, tokenizer=tokenizer, data_format=args.data_format)
    
    # 测试结果
    results = {
        "config": {
            "base_model_path": args.base_model_path,
            "medusa_weights": args.medusa_weights,
            "medusa_num_heads": args.medusa_num_heads,
            "max_seq_length": args.max_seq_length,
            "moe_num_experts": args.moe_num_experts,
            "moe_top_k": args.moe_top_k,
            "temperature": args.temperature,
            "max_new_token": args.max_new_token,
            "posterior_threshold": args.posterior_threshold,
            "posterior_alpha": args.posterior_alpha,
            "num_samples": len(test_inputs)
        },
        "medusa_results": [],
        "base_model_results": [],
        "summary": {}
    }
    
    print("=" * 60)
    print("开始测试 Medusa 模型...")
    print("=" * 60)
    
    # 测试 Medusa
    medusa_total_tokens = 0
    medusa_total_time = 0.0
    medusa_total_accept_lengths = []
    medusa_all_avg_accept_per_step = []
    
    for idx, test_item in enumerate(test_inputs):
        formatted_prompt = test_item['prompt']
        # 显示输入的前50个字符（去除system prompt）
        display_text = formatted_prompt.replace("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ", "")[:50]
        print(f"\n[{idx+1}/{len(test_inputs)}] 测试 Medusa: {display_text}...")
        try:
            result = generate_with_medusa(
                model, tokenizer, formatted_prompt,
                temperature=args.temperature,
                max_new_token=args.max_new_token,
                posterior_threshold=args.posterior_threshold,
                posterior_alpha=args.posterior_alpha
            )
            
            results["medusa_results"].append({
                "sample_id": idx,
                "input_prompt": formatted_prompt,
                "num_tokens": result["num_tokens"],
                "generation_time": result["generation_time"],
                "throughput": result["throughput"],
                "avg_accept_per_step": result["avg_accept_per_step"],
                "num_steps": result["num_steps"],
                "total_steps": result.get("total_steps", result["num_steps"]),
                "total_forward_calls": result.get("total_forward_calls", result["num_steps"] + 1),
                "kv_cache_used_count": result.get("kv_cache_used_count", result.get("total_forward_calls", result["num_steps"] + 1)),
                "fallback_count": result.get("fallback_count", 0)
            })
            
            medusa_total_tokens += result["num_tokens"]
            medusa_total_time += result["generation_time"]
            medusa_all_avg_accept_per_step.append(result["avg_accept_per_step"])
            medusa_total_accept_lengths.extend(result["accept_lengths"])
            
            # 输出生成结果和KV cache统计（参考base model的格式）
            total_steps = result.get("total_steps", result["num_steps"])
            total_forward_calls = result.get("total_forward_calls", result["num_steps"] + 1)
            kv_cache_used = result.get("kv_cache_used_count", total_forward_calls)
            fallback_used = result.get("fallback_count", 0)
            avg_accept = result["avg_accept_per_step"]
            
            print(f"  生成 {result['num_tokens']} tokens, "
                  f"耗时 {result['generation_time']:.2f}s, "
                  f"吞吐量 {result['throughput']:.2f} tokens/s, "
                  f"平均接受长度 {avg_accept:.2f}")
            
            # 输出KV cache相关统计
            if total_steps > 0:
                kv_cache_rate = kv_cache_used / total_forward_calls * 100 if total_forward_calls > 0 else 0.0
                print(f"  [KV Cache] 总步数: {total_steps}, Forward调用次数: {total_forward_calls}, "
                      f"使用KV cache: {kv_cache_used}次, 使用fallback: {fallback_used}次, "
                      f"KV cache使用率: {kv_cache_rate:.1f}%")
            else:
                print(f"  [KV Cache] 只生成了1个token，无需后续步骤")
            
            # 输出性能统计信息
            perf_stats = result.get("perf_stats", {})
            if perf_stats:
                total_time = result['generation_time']
                prefill_time = perf_stats.get("prefill_time", 0.0)
                medusa_time = perf_stats.get("total_medusa_heads_time", 0.0)
                tree_time = perf_stats.get("total_tree_decoding_time", 0.0)
                gc_time = perf_stats.get("total_generate_candidates_time", 0.0)
                ep_time = perf_stats.get("total_evaluate_posterior_time", 0.0)
                ui_time = perf_stats.get("total_update_inputs_time", 0.0)
                step_overhead_time = perf_stats.get("total_step_overhead_time", 0.0)
                remaining_time = perf_stats.get("remaining_time", 0.0)
                
                print(f"  [性能分析] Prefill: {prefill_time:.3f}s ({prefill_time/total_time*100:.1f}%), "
                      f"Medusa Heads: {medusa_time:.3f}s ({medusa_time/total_time*100:.1f}%), "
                      f"Tree Decoding: {tree_time:.3f}s ({tree_time/total_time*100:.1f}%), "
                      f"Step开销: {step_overhead_time:.3f}s ({step_overhead_time/total_time*100:.1f}%), "
                      f"其他: {remaining_time:.3f}s ({remaining_time/total_time*100:.1f}%)")
                print(f"  [详细] Generate Candidates: {gc_time:.3f}s, "
                      f"Evaluate Posterior: {ep_time:.3f}s, "
                      f"Update Inputs: {ui_time:.3f}s")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results["medusa_results"].append({
                "sample_id": idx,
                "input_prompt": formatted_prompt,
                "error": str(e)
            })
    
    print("\n" + "=" * 60)
    print("开始测试 Base Model...")
    print("=" * 60)
    
    # 测试 Base Model
    base_total_tokens = 0
    base_total_time = 0.0
    
    for idx, test_item in enumerate(test_inputs):
        formatted_prompt = test_item['prompt']
        # 显示输入的前50个字符（去除system prompt）
        display_text = formatted_prompt.replace("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ", "")[:50]
        print(f"\n[{idx+1}/{len(test_inputs)}] 测试 Base Model: {display_text}...")
        try:
            result = generate_with_base_model(
                base_model, tokenizer, formatted_prompt,
                temperature=args.temperature,
                max_new_token=args.max_new_token
            )
            
            results["base_model_results"].append({
                "sample_id": idx,
                "input_prompt": formatted_prompt,
                "num_tokens": result["num_tokens"],
                "generation_time": result["generation_time"],
                "throughput": result["throughput"],
                "total_steps": result.get("total_steps", 0),
                "kv_cache_used_count": result.get("kv_cache_used_count", 0),
                "fallback_count": result.get("fallback_count", 0),
                "perf_stats": result.get("perf_stats", {})
            })
            
            base_total_tokens += result["num_tokens"]
            base_total_time += result["generation_time"]
            
            # 输出生成结果和KV cache统计（参考Medusa的格式）
            total_steps = result.get("total_steps", 0)
            kv_cache_used = result.get("kv_cache_used_count", 0)
            fallback_used = result.get("fallback_count", 0)
            
            print(f"  生成 {result['num_tokens']} tokens, "
                  f"耗时 {result['generation_time']:.2f}s, "
                  f"吞吐量 {result['throughput']:.2f} tokens/s")
            
            # 输出KV cache相关统计
            if total_steps > 0:
                kv_cache_rate = kv_cache_used / total_steps * 100 if total_steps > 0 else 0.0
                print(f"  [KV Cache] 总步数: {total_steps}, 使用KV cache: {kv_cache_used}步, "
                      f"使用fallback: {fallback_used}步, KV cache使用率: {kv_cache_rate:.1f}%")
            else:
                print(f"  [KV Cache] 只生成了1个token，无需后续步骤")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results["base_model_results"].append({
                "sample_id": idx,
                "input_prompt": formatted_prompt,
                "error": str(e)
            })
    
    # 计算总体统计
    medusa_avg_throughput = medusa_total_tokens / medusa_total_time if medusa_total_time > 0 else 0.0
    # 总体平均接受长度 = 所有step的接受长度的平均值
    medusa_avg_accept_per_step = sum(medusa_total_accept_lengths) / len(medusa_total_accept_lengths) if medusa_total_accept_lengths else 0.0
    
    base_avg_throughput = base_total_tokens / base_total_time if base_total_time > 0 else 0.0
    
    results["summary"] = {
        "medusa": {
            "total_tokens": medusa_total_tokens,
            "total_time": medusa_total_time,
            "avg_throughput": medusa_avg_throughput,
            "avg_accept_per_step": medusa_avg_accept_per_step,
            "num_samples": len([r for r in results["medusa_results"] if "error" not in r])
        },
        "base_model": {
            "total_tokens": base_total_tokens,
            "total_time": base_total_time,
            "avg_throughput": base_avg_throughput,
            "num_samples": len([r for r in results["base_model_results"] if "error" not in r])
        },
        "speedup": medusa_avg_throughput / base_avg_throughput if base_avg_throughput > 0 else 0.0
    }
    
    # 保存结果
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"\nMedusa 模型统计:")
    print(f"  总生成token数: {medusa_total_tokens}")
    print(f"  总耗时: {medusa_total_time:.2f}s")
    print(f"  平均吞吐量: {medusa_avg_throughput:.2f} tokens/s")
    print(f"  平均每个step接受的token数: {medusa_avg_accept_per_step:.2f}")
    print(f"\nBase Model 统计:")
    print(f"  总生成token数: {base_total_tokens}")
    print(f"  总耗时: {base_total_time:.2f}s")
    print(f"  平均吞吐量: {base_avg_throughput:.2f} tokens/s")
    print(f"\n加速比: {results['summary']['speedup']:.2f}x")
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()

