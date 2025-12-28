#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Typical Acceptance策略的文本生成脚本
使用backbone模型和medusa_checkpoints_1218目录中训练好的medusa heads进行生成
"""

"""
交互模式
python t_a.py --base_model_path ../Medusa/vicuna-7b-v1.3 \
              --jittor_base_weights ./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl \
              --medusa_weights ../repro/medusa_checkpoints_1218/checkpoint-final/medusa_lm_head.jtr
"""

"""
单次生成模式

python t_a.py --input_text "你的问题" [其他参数...]
"""

import os
import sys
import json
import argparse

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
from medusa.model.modeling_medusa import MedusaModel, MedusaConfig
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig


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
            # 支持字典格式 {"role": ..., "content": ...}
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg["content"]
            else:
                # 兼容元组格式 (role, content)
                role, content = msg
            
            if role == self.roles[0]:
                ret += role + ": " + content + self.sep
            else:
                ret += role + ": " + content + self.sep2
        return ret


def load_model_and_tokenizer(base_model_path, jittor_base_weights, medusa_weights_path, medusa_num_heads=5):
    """
    加载backbone模型、medusa heads和tokenizer
    
    Args:
        base_model_path: HuggingFace模型路径（用于加载config和tokenizer）
        jittor_base_weights: Jittor格式的backbone模型权重路径
        medusa_weights_path: Medusa heads权重路径
        medusa_num_heads: Medusa heads数量
    
    Returns:
        model: 加载好的MedusaModel
        tokenizer: tokenizer
    """
    print("=" * 60)
    print("正在加载模型...")
    print("=" * 60)
    
    # 1. 加载模型配置
    print(f"1. 加载配置从: {base_model_path}")
    with open(os.path.join(base_model_path, "config.json"), 'r') as f:
        config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(config_dict)
    
    # 2. 加载 Tokenizer
    print("2. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    
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
        base_model_name_or_path=base_model_path
    )
    model = MedusaModel(medusa_config, base_model=base_model)
    
    if medusa_weights_path and os.path.exists(medusa_weights_path):
        print(f"   从 {medusa_weights_path} 加载 Medusa heads 权重...")
        medusa_weights = jt.load(medusa_weights_path)
        model.medusa_head.load_parameters(medusa_weights)
    
    model.eval()
    
    print("模型加载完成！")
    print("=" * 60)
    
    return model, tokenizer


def generate_with_typical_acceptance(model, tokenizer, input_text, 
                                     temperature=0.7,
                                     max_new_token=512,
                                     posterior_threshold=0.01,
                                     posterior_alpha=0.1,
                                     medusa_choices=None):
    """
    使用Typical Acceptance策略进行文本生成
    
    Typical Acceptance策略原理：
    1. 第一个token使用贪婪解码，无条件接受
    2. 后续tokens使用typical acceptance机制判断是否接受
    3. 接受条件：p_original > min(epsilon, delta * exp(-H))
       其中H是熵，epsilon=posterior_threshold, delta=posterior_alpha
    
    Args:
        model: MedusaModel实例
        tokenizer: tokenizer实例
        input_text: 输入文本
        temperature: 温度参数
        max_new_token: 最大生成token数
        posterior_threshold: epsilon阈值（对应公式中的epsilon）
        posterior_alpha: delta参数（对应公式中的delta）
        medusa_choices: Medusa树结构选择，None则使用默认值
    
    Returns:
        output_text: 生成的文本
    """
    # 使用对话格式格式化输入
    conv = SimpleConversation()
    conv.append_message(conv.roles[0], input_text)  # USER: ...
    conv.append_message(conv.roles[1], None)  # ASSISTANT: (等待生成)
    formatted_prompt = conv.get_prompt()
    
    # 编码输入（使用格式化后的prompt）
    input_ids = tokenizer([formatted_prompt], return_tensors="np")["input_ids"]
    input_ids = jt.array(input_ids)
    
    input_len = input_ids.shape[1]
    
    print(f"\n用户输入: {input_text}")
    print(f"格式化后的Prompt长度: {input_len} tokens")
    print(f"开始生成（Typical Acceptance策略）...")
    print("-" * 60)
    
    # 使用medusa_generate进行生成
    # sampling='typical' 表示使用typical acceptance策略
    # fast=True 使用快速模式（基于posterior概率的快速评估）
    with jt.no_grad():
        generated_text = ""
        generated_ids = []
        step_count = 0
        accept_lengths = []  # 收集每个step的accept_length
        
        # 保存输入的token IDs，用于后续提取新生成的部分
        input_ids_list = input_ids[0].numpy().tolist()
        
        generator = model.medusa_generate(
            input_ids,
            temperature=temperature,
            max_steps=max_new_token,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            medusa_choices=medusa_choices,
            sampling='typical',  # 使用typical acceptance策略
            fast=True,  # 使用快速模式
            tokenizer=tokenizer,
            debug_callback=debug_callback_first_head  # 输出第一个medusa head的调试信息
        )
        
        for output in generator:
            step_count += 1
            if "text" in output:
                generated_text = output["text"]
            if "ids" in output:
                generated_ids = output["ids"]
            if "accept_length" in output:
                # accept_length是medusa预测的token中被接受的数量
                # 每个step实际接受的token数 = accept_length + 1 (base model的1个token)
                accept_lengths.append(output["accept_length"] + 1)
    
    print(f"\n生成完成！共 {step_count} 步")
    print("-" * 60)
    
    # 计算平均接受的token数量
    if accept_lengths:
        avg_accept_tokens = sum(accept_lengths) / len(accept_lengths)
        print(f"每个step平均接受的token数量: {avg_accept_tokens:.2f}")
    else:
        print(f"每个step平均接受的token数量: 1.00 (未收集到数据，使用默认值)")
    print("-" * 60)
    
    # 提取新生成的部分（使用token IDs，更准确）
    # 注意：generated_ids 已经是新生成的token了（medusa_generate返回的是input_ids[input_len:]）
    # 所以不需要再减去输入的长度
    if generated_ids:
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        # 备用方法：使用字符串长度（可能不准确，但作为备用）
        if len(generated_text) > len(input_text):
            output_text = generated_text[len(input_text):].strip()
        else:
            output_text = generated_text
    
    return output_text


def escape_special_chars(text, max_len=20):
    """
    转义特殊字符，将不可打印的字符转换为转义序列
    
    Args:
        text: 要转义的文本
        max_len: 最大显示长度
    
    Returns:
        转义后的文本
    """
    # 使用repr()来转义特殊字符，然后去掉两端的引号
    # repr()会将特殊字符转换为转义序列，如'\n' -> '\\n'
    escaped = repr(text)
    # 去掉两端的引号（repr()会在两端添加引号）
    if (escaped.startswith("'") and escaped.endswith("'")) or \
       (escaped.startswith('"') and escaped.endswith('"')):
        escaped = escaped[1:-1]
    
    # 限制长度
    if len(escaped) > max_len:
        escaped = escaped[:max_len-3] + "..."
    
    return escaped


def debug_callback_first_head(debug_info, step_idx, tokenizer):
    """
    调试回调函数：显示第一个medusa head的调试信息
    
    Args:
        debug_info: 从evaluate_posterior返回的调试信息
        step_idx: 当前生成步数
        tokenizer: tokenizer实例
    """
    if debug_info is None:
        return
    
    print(f"\n[Step {step_idx}] 第一个Medusa Head的Token调试信息:")
    print("=" * 80)
    
    # 显示当前序列信息
    current_sequence_info = debug_info.get('current_sequence_info')
    if current_sequence_info is not None:
        print(f"当前序列长度: {current_sequence_info['total_length']} tokens")
        display_tokens = current_sequence_info['display_tokens']
        display_text = current_sequence_info['display_text']
        # 转义特殊字符
        display_text_escaped = escape_special_chars(display_text, max_len=100)
        print(f"当前序列（最后20个token）: {display_text_escaped}")
        print(f"Token IDs: {display_tokens}")
    else:
        print("当前序列信息: 未提供")
    
    print("-" * 80)
    
    # 显示Base Model预测的token
    base_token_info = debug_info.get('base_token_info')
    if base_token_info is not None:
        base_token_id = base_token_info['token_id']
        base_token_text = base_token_info['token_text']
        base_token_text_escaped = escape_special_chars(base_token_text, max_len=50)
        print(f"Base Model预测的Token: ID={base_token_id}, 文本='{base_token_text_escaped}'")
    else:
        base_token = debug_info.get('base_token')
        if base_token is not None:
            print(f"Base Model预测的Token: ID={base_token}")
        else:
            print("Base Model预测的Token: 未提供")
    
    print("-" * 80)
    
    # 显示第一个medusa head的token信息
    first_head_debug = debug_info.get('first_head_debug')
    if first_head_debug is None or 'tokens' not in first_head_debug:
        print("第一个Medusa Head的Token信息: 未提供")
        print("=" * 80)
        return
    
    tokens_info = first_head_debug['tokens']
    if len(tokens_info) == 0:
        print("第一个Medusa Head的Token信息: 无token")
        print("=" * 80)
        return
    
    print("第一个Medusa Head的Token列表:")
    print(f"{'Token ID':<12} {'Token文本':<20} {'p_original':<15} {'阈值':<15} {'是否接受':<10}")
    print("-" * 80)
    
    for token_info in tokens_info:
        token_id = token_info['token_id']
        p_original = token_info['p_original']
        threshold_val = token_info['threshold']
        is_accepted = token_info['is_accepted']
        
        # 尝试解码token文本
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            # 转义特殊字符，确保格式整齐
            token_text = escape_special_chars(token_text, max_len=20)
        except:
            token_text = f"<token_{token_id}>"
        
        accept_str = "✓ 接受" if is_accepted else "✗ 拒绝"
        print(f"{token_id:<12} {token_text:<20} {p_original:<15.6f} {threshold_val:<15.6f} {accept_str:<10}")
    
    print("=" * 80)


def interactive_mode(model, tokenizer, **gen_kwargs):
    """
    交互式模式：从终端读取输入，输出模型回答
    
    Args:
        model: MedusaModel实例
        tokenizer: tokenizer实例
        **gen_kwargs: 传递给generate_with_typical_acceptance的参数
    """
    print("\n" + "=" * 60)
    print("进入交互模式（Typical Acceptance策略）")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # 读取用户输入
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            # 生成回答
            response = generate_with_typical_acceptance(
                model, tokenizer, user_input, **gen_kwargs
            )
            
            # 输出回答
            print(f"\n助手: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            print("请重试...\n")


def main():
    parser = argparse.ArgumentParser(description="Typical Acceptance策略文本生成")
    
    # 路径参数
    parser.add_argument("--base_model_path", type=str, 
                       default="../Medusa/vicuna-7b-v1.3",
                       help="HuggingFace模型路径（用于config和tokenizer）")
    parser.add_argument("--jittor_base_weights", type=str,
                       default="./vicuna-jittor-weights/vicuna-7b-v1.3_f16.pkl",
                       help="Jittor格式的backbone模型权重")
    parser.add_argument("--medusa_weights", type=str,
                       default="./medusa_checkpoints_1218/checkpoint-best/medusa_lm_head.jtr",
                       help="Medusa heads权重路径")
    
    # 模型参数
    parser.add_argument("--medusa_num_heads", type=int, default=5,
                       help="Medusa heads数量")
    
    # 生成参数（Typical Acceptance相关）
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数，控制生成的随机性")
    parser.add_argument("--max_new_token", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--posterior_threshold", type=float, default=0.01,
                       help="Typical Acceptance的epsilon阈值")
    parser.add_argument("--posterior_alpha", type=float, default=0.1,
                       help="Typical Acceptance的delta参数")
    parser.add_argument("--medusa_choices", type=str, default=None,
                       help="Medusa树结构，None则使用默认值")
    
    # 单次生成模式（非交互）
    parser.add_argument("--input_text", type=str, default=None,
                       help="单次生成模式的输入文本（如果提供，则不进入交互模式）")
    
    args = parser.parse_args()
    
    # 解析medusa_choices（如果是字符串）
    medusa_choices = None
    if args.medusa_choices:
        try:
            medusa_choices = eval(args.medusa_choices)
        except:
            print(f"警告: 无法解析medusa_choices，使用默认值")
            medusa_choices = None
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        args.base_model_path,
        args.jittor_base_weights,
        args.medusa_weights,
        args.medusa_num_heads
    )
    
    # 准备生成参数
    gen_kwargs = {
        'temperature': args.temperature,
        'max_new_token': args.max_new_token,
        'posterior_threshold': args.posterior_threshold,
        'posterior_alpha': args.posterior_alpha,
        'medusa_choices': medusa_choices
    }
    
    # Warmup（预热Jittor编译）
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
        sys.exit(1)
    
    # 运行生成
    if args.input_text:
        # 单次生成模式
        response = generate_with_typical_acceptance(
            model, tokenizer, args.input_text, **gen_kwargs
        )
        print(f"\n生成的回答:\n{response}\n")
    else:
        # 交互模式
        interactive_mode(model, tokenizer, **gen_kwargs)


if __name__ == "__main__":
    main()

