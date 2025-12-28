#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Model 基准测试脚本
测试 Base Model 在测试数据上的性能
使用如下方式测试
python test_base_model_benchmark.py \
    --base_model_path ../Medusa/vicuna-7b-v1.3 \
    --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
    --data_path ../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --num_samples 20 \
    --max_new_token 4096
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
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig


def load_base_model_and_tokenizer(base_model_path, jittor_base_weights, max_seq_length=4096):
    """
    加载base model和tokenizer
    
    Args:
        base_model_path: HuggingFace模型路径
        jittor_base_weights: Jittor格式的backbone模型权重路径
        max_seq_length: 最大序列长度（默认4096，可选4096或8192）
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
    
    print("模型加载完成！")
    print("=" * 60)
    
    return base_model, tokenizer


def _validate_past_key_values(past_key_values):
    """
    验证past_key_values的结构是否正确
    
    Args:
        past_key_values: 要验证的past_key_values
        
    Returns:
        如果结构正确返回True，否则返回False
    """
    if past_key_values is None:
        return False
    
    try:
        # 检查是否是tuple或list
        if not isinstance(past_key_values, (tuple, list)):
            return False
        
        # 检查是否为空
        if len(past_key_values) == 0:
            return False
        
        # 检查第一层结构
        first_layer = past_key_values[0]
        if not isinstance(first_layer, (tuple, list)):
            return False
        
        # 检查第二层结构（应该是(k, v)对）
        if len(first_layer) < 2:
            return False
        
        # 检查k和v是否都是有效的tensor（不是None）
        k_tensor = first_layer[0]
        v_tensor = first_layer[1]
        
        if k_tensor is None or v_tensor is None:
            return False
        
        # 检查是否是jt.Var类型
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
    
    with jt.no_grad():
        generated_ids = []
        past_key_values = None
        accumulated_input_ids = input_ids  # 用于fallback方案
        kv_cache_used_count = 0  # 统计使用KV cache的次数
        fallback_count = 0  # 统计使用fallback的次数
        
        # 第一次forward：传入完整的input_ids
        outputs = base_model(
            input_ids=input_ids,
            past_key_values=None,  # 第一次调用时明确传入None
            use_cache=True,
            return_dict=True
        )
        
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
        if temperature == 0:
            next_token_id = jt.argmax(next_token_logits, dim=-1).item()
        else:
            # 使用温度采样
            probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
            next_token_idx = jt.multinomial(probs, 1)
            next_token_id = next_token_idx.item()
        
        generated_ids.append(next_token_id)
        
        # 检查是否遇到EOS token
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
                "total_steps": 0
            }
        
        # 后续生成：只传入新生成的token
        for step in range(1, max_new_token):
            # 准备下一个token的input_ids（只有新生成的token）
            next_token = jt.array([[next_token_id]], dtype="int64")
            
            # 如果past_key_values无效，使用累积的input_ids（fallback方案）
            kv_cache_valid = _validate_past_key_values(past_key_values)
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
            
            # 处理返回值
            if isinstance(outputs, tuple):
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            
            # 验证past_key_values的结构
            if not _validate_past_key_values(past_key_values):
                past_key_values = None
            
            # 检查logits是否有效
            if logits is None:
                raise ValueError(f"Model returned None logits at step {step}")
            
            # 获取logits（只有一个位置）
            next_token_logits = logits[0, -1, :]
            
            # 采样下一个token
            if temperature == 0:
                next_token_id = jt.argmax(next_token_logits, dim=-1).item()
            else:
                # 使用温度采样
                probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
                next_token_idx = jt.multinomial(probs, 1)
                next_token_id = next_token_idx.item()
            
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
    
    return {
        "generated_ids": generated_ids,
        "num_tokens": num_generated_tokens,
        "generation_time": generation_time,
        "throughput": throughput,
        "kv_cache_used_count": kv_cache_used_count,
        "fallback_count": fallback_count,
        "total_steps": total_steps
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


def load_test_data(data_path, num_samples=200, tokenizer=None):
    """
    加载测试数据（前num_samples条）
    按照训练时的格式处理数据
    """
    print(f"正在加载测试数据: {data_path}")
    
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
        test_data = []
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
    parser = argparse.ArgumentParser(description="Base Model 基准测试")
    
    # 路径参数
    parser.add_argument("--base_model_path", type=str, 
                       default="../Medusa/vicuna-7b-v1.3",
                       help="HuggingFace模型路径")
    parser.add_argument("--jittor_base_weights", type=str,
                       default="./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl",
                       help="Jittor格式的backbone模型权重")
    parser.add_argument("--data_path", type=str,
                       default="../Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
                       help="测试数据路径")
    
    # 模型参数
    parser.add_argument("--max_seq_length", type=int, default=4096,
                       choices=[4096, 8192],
                       help="最大序列长度（4096或8192，默认4096）")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    parser.add_argument("--max_new_token", type=int, default=512,
                       help="最大生成token数")
    
    # 测试参数
    parser.add_argument("--num_samples", type=int, default=200,
                       help="测试样本数量")
    
    args = parser.parse_args()
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 加载模型
    base_model, tokenizer = load_base_model_and_tokenizer(
        args.base_model_path,
        args.jittor_base_weights,
        max_seq_length=args.max_seq_length
    )
    
    # Warmup
    print("\n正在预热模型...")
    try:
        dummy_input = jt.array(tokenizer(["Hello"], return_tensors="np")["input_ids"])
        with jt.no_grad():
            _ = base_model(dummy_input, use_cache=True, return_dict=True)
        jt.sync_all()
        jt.gc()
        print("预热完成！\n")
    except Exception as e:
        print(f"预热失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 加载测试数据（需要tokenizer来格式化）
    test_inputs = load_test_data(args.data_path, args.num_samples, tokenizer=tokenizer)
    
    # 测试结果
    results = {
        "config": {
            "base_model_path": args.base_model_path,
            "max_seq_length": args.max_seq_length,
            "temperature": args.temperature,
            "max_new_token": args.max_new_token,
            "num_samples": len(test_inputs)
        },
        "base_model_results": [],
        "summary": {}
    }
    
    print("=" * 60)
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
                "fallback_count": result.get("fallback_count", 0)
            })
            
            base_total_tokens += result["num_tokens"]
            base_total_time += result["generation_time"]
            
            # 输出生成结果和KV cache统计
            total_steps = result.get("total_steps", 0)
            kv_cache_used = result.get("kv_cache_used_count", 0)
            fallback_used = result.get("fallback_count", 0)
            
            print(f"  生成 {result['num_tokens']} tokens, "
                  f"耗时 {result['generation_time']:.2f}s, "
                  f"吞吐量 {result['throughput']:.2f} tokens/s")
            
            if total_steps > 0:
                kv_cache_rate = kv_cache_used / total_steps * 100
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
    base_avg_throughput = base_total_tokens / base_total_time if base_total_time > 0 else 0.0
    
    results["summary"] = {
        "base_model": {
            "total_tokens": base_total_tokens,
            "total_time": base_total_time,
            "avg_throughput": base_avg_throughput,
            "num_samples": len([r for r in results["base_model_results"] if "error" not in r])
        }
    }
    
    # 保存结果（使用不同的文件名，避免覆盖之前的测试结果）
    results_file = results_dir / "base_model_benchmark_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"\nBase Model 统计:")
    print(f"  总生成token数: {base_total_tokens}")
    print(f"  总耗时: {base_total_time:.2f}s")
    print(f"  平均吞吐量: {base_avg_throughput:.2f} tokens/s")
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()

