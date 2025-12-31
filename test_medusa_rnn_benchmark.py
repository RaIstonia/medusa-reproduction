#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medusa (Gated RNN Version) 模型基准测试脚本
测试 Medusa 和 Base Model 在相同数据上的性能对比

使用ShareGPT格式数据测试：
python test_medusa_rnn_benchmark.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ../repro/medusa_checkpoints_rnn_1230/checkpoint-final/medusa_lm_head.jtr \
    --data_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096

使用MT-bench数据集测试：
python test_medusa_rnn_benchmark.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --medusa_weights ../repro/medusa_checkpoints_rnn_1230/checkpoint-final/medusa_lm_head.jtr \
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

# [修改点 1] 导入路径改为 model_rnn
from medusa.model_rnn.modeling_medusa import MedusaModel, MedusaConfig
from medusa.model_rnn.modeling_llama import LlamaForCausalLM, LlamaConfig


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


def load_model_and_tokenizer(base_model_path, jittor_base_weights, medusa_weights_path, medusa_num_heads=5, max_seq_length=4096):
    """
    加载backbone模型、medusa heads和tokenizer
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
    print("4. 加载 Medusa Model (Gated RNN)...")
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=1,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size,
        base_model_name_or_path=base_model_path
    )
    model = MedusaModel(medusa_config, base_model=base_model)
    
    # [修改点 2] 使用 load_medusa_weights 方法加载权重
    # 因为新模型的属性名是 medusa_blocks 而非 medusa_head
    if medusa_weights_path and os.path.exists(medusa_weights_path):
        model.load_medusa_weights(medusa_weights_path)
    
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
    """
    input_ids = tokenizer([formatted_prompt], return_tensors="np", add_special_tokens=False)["input_ids"]
    input_ids = jt.array(input_ids)
    
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
        
        # 初始化统计字典
        perf_stats = {}
        
        for output in generator:
            step_count += 1
            if "ids" in output:
                generated_ids = output["ids"]
            if "accept_length" in output:
                accept_lengths.append(output["accept_length"] + 1)
            
            # 更新性能统计信息
            if "total_medusa_heads_time" in output:
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
    
    num_generated_tokens = len(generated_ids) if generated_ids else 0
    throughput = num_generated_tokens / generation_time if generation_time > 0 else 0.0
    avg_accept_per_step = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0
    
    total_steps = step_count
    total_forward_calls = step_count + 1
    kv_cache_used_count = total_forward_calls
    fallback_count = 0
    
    # 确保 perf_stats 有值
    if not perf_stats:
        perf_stats = {
            "prefill_time": 0.0,
            "total_medusa_heads_time": 0.0,
            "total_tree_decoding_time": 0.0,
            "total_generate_candidates_time": 0.0,
            "total_evaluate_posterior_time": 0.0,
            "total_update_inputs_time": 0.0,
            "total_step_overhead_time": 0.0,
        }
    
    measured_time = sum(perf_stats.values())
    remaining_time = max(0.0, generation_time - measured_time)
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
        "perf_stats": perf_stats
    }


def _validate_past_key_values(past_key_values):
    """
    验证past_key_values的结构是否正确
    """
    if past_key_values is None:
        return False
    try:
        if not isinstance(past_key_values, (tuple, list)):
            return False
        for layer_past in past_key_values:
            if not isinstance(layer_past, (tuple, list)) or len(layer_past) < 2:
                return False
            k_tensor, v_tensor = layer_past[0], layer_past[1]
            if not isinstance(k_tensor, jt.Var) or not isinstance(v_tensor, jt.Var):
                return False
            if len(k_tensor.shape) < 3 or len(v_tensor.shape) < 3:
                return False
        return True
    except (TypeError, IndexError, AttributeError):
        return False


def generate_with_base_model(base_model, tokenizer, formatted_prompt,
                             temperature=0.7,
                             max_new_token=512):
    """
    使用 Base Model 进行生成
    """
    input_ids = tokenizer([formatted_prompt], return_tensors="np", add_special_tokens=False)["input_ids"]
    input_ids = jt.array(input_ids)
    
    start_time = time.time()
    
    prefill_time = 0.0
    decoding_forward_time = 0.0
    sampling_time = 0.0
    other_time = 0.0
    
    with jt.no_grad():
        generated_ids = []
        past_key_values = None
        accumulated_input_ids = input_ids
        kv_cache_used_count = 0
        fallback_count = 0
        
        # Prefill
        prefill_start = time.time()
        outputs = base_model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True
        )
        jt.sync_all()
        prefill_time = time.time() - prefill_start
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
        else:
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        
        if not _validate_past_key_values(past_key_values):
            past_key_values = None
        
        next_token_logits = logits[0, -1, :]
        
        # First Sampling
        sampling_start = time.time()
        if temperature == 0:
            next_token_id = jt.argmax(next_token_logits, dim=-1).item()
        else:
            probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
            next_token_idx = jt.multinomial(probs, 1)
            next_token_id = next_token_idx.item()
        sampling_time += time.time() - sampling_start
        
        generated_ids.append(next_token_id)
        
        if next_token_id == tokenizer.eos_token_id:
            end_time = time.time()
            generation_time = end_time - start_time
            num_generated_tokens = len(generated_ids)
            throughput = num_generated_tokens / generation_time if generation_time > 0 else 0.0
            return {
                "generated_ids": generated_ids,
                "num_tokens": num_generated_tokens,
                "generation_time": generation_time,
                "throughput": throughput,
                "kv_cache_used_count": 0,
                "fallback_count": 0,
                "total_steps": 0,
                "perf_stats": {"prefill_time": prefill_time}
            }
        
        # Decoding Loop
        for step in range(1, max_new_token):
            next_token = jt.array([[next_token_id]], dtype="int64")
            
            kv_cache_valid = _validate_past_key_values(past_key_values)
            forward_start = time.time()
            if not kv_cache_valid:
                fallback_count += 1
                accumulated_input_ids = jt.concat([accumulated_input_ids, next_token], dim=1)
                outputs = base_model(
                    input_ids=accumulated_input_ids,
                    past_key_values=None,
                    use_cache=True,
                    return_dict=True
                )
            else:
                kv_cache_used_count += 1
                outputs = base_model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            jt.sync_all()
            decoding_forward_time += time.time() - forward_start
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            
            if not _validate_past_key_values(past_key_values):
                past_key_values = None
            
            next_token_logits = logits[0, -1, :]
            
            sampling_start = time.time()
            if temperature == 0:
                next_token_id = jt.argmax(next_token_logits, dim=-1).item()
            else:
                probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
                next_token_idx = jt.multinomial(probs, 1)
                next_token_id = next_token_idx.item()
            sampling_time += time.time() - sampling_start
            
            generated_ids.append(next_token_id)
            
            if next_token_id == tokenizer.eos_token_id:
                break
        
        total_steps = len(generated_ids) - 1
    
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
    格式化对话
    """
    separator = " "
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    last_gpt_idx = -1
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("from") == "gpt":
            last_gpt_idx = i
            break
    
    formatted_parts = [system_prompt + separator]
    
    for i, turn in enumerate(conversations):
        role = turn.get("from", "")
        content = turn.get("value", "")
        
        if role == "human":
            role_str = "USER: "
            formatted_parts.append(role_str + content + separator)
        elif role == "gpt":
            role_str = "ASSISTANT: "
            if i == last_gpt_idx:
                formatted_parts.append(role_str)
            else:
                formatted_parts.append(role_str + content + eos_token)
    
    full_prompt = "".join(formatted_parts)
    input_ids = tokenizer([full_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
    
    if len(input_ids) > max_length:
        last_human_idx = -1
        for i in range(len(conversations) - 1, -1, -1):
            if conversations[i].get("from") == "human":
                last_human_idx = i
                break
        
        if last_human_idx >= 0:
            last_human_content = conversations[last_human_idx].get("value", "")
            min_prompt = system_prompt + separator + "USER: " + last_human_content + separator + "ASSISTANT: "
            min_ids = tokenizer([min_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
            
            if len(min_ids) > max_length:
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
                    full_prompt = system_prompt + separator + "ASSISTANT: "
            else:
                remaining_length = max_length - len(min_ids)
                history_parts = []
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
                    
                    test_parts = [system_prompt + separator] + history_parts + [part] + ["USER: " + last_human_content + separator + "ASSISTANT: "]
                    test_prompt = "".join(test_parts)
                    test_ids = tokenizer([test_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
                    
                    if len(test_ids) <= max_length:
                        history_parts.append(part)
                    else:
                        break
                full_prompt = system_prompt + separator + "".join(history_parts) + "USER: " + last_human_content + separator + "ASSISTANT: "
        else:
            full_prompt = system_prompt + separator + "ASSISTANT: "
        
        final_ids = tokenizer([full_prompt], return_tensors="np", add_special_tokens=False)["input_ids"][0]
        if len(final_ids) > max_length:
            truncated_ids = final_ids[:max_length]
            full_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            if not full_prompt.rstrip().endswith("ASSISTANT:"):
                last_assistant_pos = full_prompt.rfind("ASSISTANT:")
                if last_assistant_pos >= 0:
                    full_prompt = full_prompt[:last_assistant_pos + len("ASSISTANT:")]
                else:
                    full_prompt = system_prompt + separator + "ASSISTANT: "
    
    return full_prompt


def load_test_data(data_path, num_samples=200, tokenizer=None, data_format='auto'):
    """
    加载测试数据
    """
    print(f"正在加载测试数据: {data_path}")
    
    if data_format == 'auto':
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
                        f.seek(0)
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            if 'conversations' in data[0]:
                                data_format = 'sharegpt'
                                print("检测到ShareGPT格式（JSON）")
                            else:
                                data_format = 'sharegpt'
                                print("假设为ShareGPT格式")
                        else:
                            data_format = 'sharegpt'
        except Exception as e:
            print(f"自动检测格式失败: {e}，假设为ShareGPT格式")
            data_format = 'sharegpt'
    
    test_data = []
    
    if data_format == 'mt_bench':
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
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                test_data = data[:num_samples]
            else:
                test_data = list(data.values())[:num_samples] if isinstance(data, dict) else [data][:num_samples]
        except (json.JSONDecodeError, MemoryError):
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
    
    test_inputs = []
    for item in test_data:
        if data_format == 'mt_bench':
            turns = item.get('turns', [])
            if len(turns) > 0:
                first_conversations = [
                    {"from": "human", "value": turns[0]},
                    {"from": "gpt", "value": ""}
                ]
                max_length = getattr(tokenizer, 'model_max_length', 4096) if tokenizer else 4096
                formatted_prompt = format_conversation_for_inference(first_conversations, tokenizer, max_length=max_length)
                test_inputs.append({
                    'prompt': formatted_prompt,
                    'conversations': first_conversations,
                    'question_id': item.get('question_id'),
                    'category': item.get('category'),
                    'turn': 1,
                    'original_item': item
                })
        else:
            conversations = item.get('conversations', [])
            if conversations and len(conversations) > 0:
                has_human = any(turn.get('from') == 'human' for turn in conversations)
                if has_human:
                    max_length = getattr(tokenizer, 'model_max_length', 4096) if tokenizer else 4096
                    formatted_prompt = format_conversation_for_inference(conversations, tokenizer, max_length=max_length)
                    test_inputs.append({
                        'prompt': formatted_prompt,
                        'conversations': conversations
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
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    model, base_model, tokenizer = load_model_and_tokenizer(
        args.base_model_path,
        args.jittor_base_weights,
        args.medusa_weights,
        args.medusa_num_heads,
        max_seq_length=args.max_seq_length
    )
    
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
    
    test_inputs = load_test_data(args.data_path, args.num_samples, tokenizer=tokenizer, data_format=args.data_format)
    
    results = {
        "config": {
            "base_model_path": args.base_model_path,
            "medusa_weights": args.medusa_weights,
            "medusa_num_heads": args.medusa_num_heads,
            "max_seq_length": args.max_seq_length,
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
    
    medusa_total_tokens = 0
    medusa_total_time = 0.0
    medusa_total_accept_lengths = []
    
    for idx, test_item in enumerate(test_inputs):
        formatted_prompt = test_item['prompt']
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
            medusa_total_accept_lengths.extend(result["accept_lengths"])
            
            total_steps = result.get("total_steps", result["num_steps"])
            total_forward_calls = result.get("total_forward_calls", result["num_steps"] + 1)
            kv_cache_used = result.get("kv_cache_used_count", total_forward_calls)
            avg_accept = result["avg_accept_per_step"]
            
            print(f"  生成 {result['num_tokens']} tokens, "
                  f"耗时 {result['generation_time']:.2f}s, "
                  f"吞吐量 {result['throughput']:.2f} tokens/s, "
                  f"平均接受长度 {avg_accept:.2f}")
            
            perf_stats = result.get("perf_stats", {})
            if perf_stats:
                total_time = result['generation_time']
                prefill_time = perf_stats.get("prefill_time", 0.0)
                medusa_time = perf_stats.get("total_medusa_heads_time", 0.0)
                tree_time = perf_stats.get("total_tree_decoding_time", 0.0)
                step_overhead_time = perf_stats.get("total_step_overhead_time", 0.0)
                remaining_time = perf_stats.get("remaining_time", 0.0)
                
                print(f"  [性能分析] Prefill: {prefill_time:.3f}s, "
                      f"Medusa: {medusa_time:.3f}s, "
                      f"Tree: {tree_time:.3f}s")
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
    
    base_total_tokens = 0
    base_total_time = 0.0
    
    for idx, test_item in enumerate(test_inputs):
        formatted_prompt = test_item['prompt']
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
            
            print(f"  生成 {result['num_tokens']} tokens, "
                  f"耗时 {result['generation_time']:.2f}s, "
                  f"吞吐量 {result['throughput']:.2f} tokens/s")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results["base_model_results"].append({
                "sample_id": idx,
                "input_prompt": formatted_prompt,
                "error": str(e)
            })
    
    medusa_avg_throughput = medusa_total_tokens / medusa_total_time if medusa_total_time > 0 else 0.0
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