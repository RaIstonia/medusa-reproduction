import argparse
import json
import os
import time
import uuid
import numpy as np
from tqdm import tqdm

# --- [Hack] 解决 PyTorch 和 Jittor 的库冲突 ---
# 在导入 torch 之前隐藏 GPU，防止 PyTorch 加载 CUDA 库
# 保存原始值以便恢复
_original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import torch
# 依赖 HuggingFace transformers 进行 tokenizer 处理
from transformers import AutoTokenizer
# 恢复 GPU 可见性
if _original_cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _original_cuda_visible_devices
else:
    # 如果原本没有设置，则删除
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

# --- Jittor 导入 ---
import jittor as jt
# 开启 CUDA
jt.flags.use_cuda = 1

# 导入你的模型定义
# 假设你的文件结构是 medusa/model/modeling_medusa.py 和 medusa/model/modeling_llama.py
from medusa.model.modeling_medusa import MedusaModel, MedusaConfig
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig

def get_short_uuid():
    """生成简短的UUID，替代 shortuuid 库"""
    return str(uuid.uuid4())[:8]

def load_questions(question_file, begin, end):
    """加载问题列表"""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

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

def medusa_inference(
    model, 
    tokenizer, 
    input_ids, 
    args
):
    """
    执行单次推理，并统计时间与步数
    """
    # === [防御 1] 词表越界修正 ===
    vocab_size = model.config.vocab_size
    
    # 先转换为 numpy 进行越界检查
    if isinstance(input_ids, jt.Var):
        input_ids_np = input_ids.numpy()
    else:
        input_ids_np = np.array(input_ids)
    
    # 检查并修正越界的 token ID
    if np.any(input_ids_np >= vocab_size) or np.any(input_ids_np < 0):
        print(f"[Warning] Found out-of-vocab token IDs. Clipping to [0, {vocab_size-1}]")
        input_ids_np = np.clip(input_ids_np, 0, vocab_size - 1)
    
    # 转换回 Jittor 变量
    input_ids = jt.array(input_ids_np)
    
    # === [防御 2] 长度检查 ===
    MAX_MODEL_LEN = 2048
    if hasattr(model, 'base_model') and model.base_model and hasattr(model.base_model, 'config'):
        MAX_MODEL_LEN = getattr(model.base_model.config, 'max_position_embeddings', 2048)
    
    # Medusa 树形探测需要安全余量
    MEDUSA_BUFFER = 64  # 比之前更大的缓冲区
    max_safe_input_len = MAX_MODEL_LEN - MEDUSA_BUFFER
    
    current_len = input_ids.shape[1]
    if current_len > max_safe_input_len:
        print(f"[Warning] Prompt too long ({current_len}). Truncating to {max_safe_input_len}.")
        input_ids = input_ids[:, -max_safe_input_len:]
    
    # 动态调整 max_new_token
    current_len = input_ids.shape[1]
    space_left = MAX_MODEL_LEN - current_len - MEDUSA_BUFFER
    real_max_new_token = max(1, min(args.max_new_token, space_left))
    
    input_len = input_ids.shape[1]
    
    # 同步以确保计时准确，并捕获之前的异步错误
    jt.sync_all()
    start_time = time.time()
    
    step_count = 0
    generated_ids = []
    output_text = ""
    
    # 调用生成器
    with jt.no_grad():
        # model.medusa_generate 是一个生成器，我们需要迭代它
        generator = model.medusa_generate(
            input_ids,
            temperature=args.temperature,
            max_steps=real_max_new_token,  # 使用计算后的安全长度
            posterior_threshold=args.posterior_threshold,
            posterior_alpha=args.posterior_alpha,
            medusa_choices=args.medusa_choices,
            tokenizer=tokenizer # 传入 tokenizer 以便内部解码和判断 EOS
        )
        
        for output in generator:
            step_count += 1
            # 获取当前最新的结果
            if "ids" in output:
                generated_ids = output["ids"]
            if "text" in output:
                output_text = output["text"]
    
    # 再次同步结束计时
    jt.sync_all()
    wall_time = time.time() - start_time
    
    # 清理缓存，防止内存泄漏
    jt.gc()
    
    # 计算生成的 token 数量 (去除 prompt 部分)
    # generated_ids 可能是 list 或 np.array
    total_len = len(generated_ids) if isinstance(generated_ids, list) else generated_ids.shape[0]
    new_tokens_count = total_len - input_len
    if new_tokens_count < 0: new_tokens_count = 0 # 防止异常

    return output_text, new_tokens_count, wall_time, step_count

def run_eval(args):
    # 1. 加载模型配置
    print(f"Loading base config from {args.base_model_path}...")
    with open(os.path.join(args.base_model_path, "config.json"), 'r') as f:
        config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(config_dict)
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    
    # 3. 加载基座模型
    print("Loading Base Model...")
    base_model = LlamaForCausalLM(llama_config)
    if args.jittor_base_weights:
        base_weights = jt.load(args.jittor_base_weights)
        # 确保转换为 FP16
        base_model.load_parameters(base_weights)
    base_model.float16()
    base_model.eval()
    
    # 4. 加载 Medusa
    print("Loading Medusa Model...")
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=1,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size,
        base_model_name_or_path=args.base_model_path
    )
    model = MedusaModel(medusa_config, base_model=base_model)
    
    if args.medusa_weights:
        print(f"Loading Medusa weights from {args.medusa_weights}...")
        medusa_weights = jt.load(args.medusa_weights)
        model.medusa_head.load_parameters(medusa_weights)
    
    model.eval()
    
    # 解析 Medusa Choices (如果是字符串)
    if isinstance(args.medusa_choices, str):
        # 注意：这里假设输入是类似 "[[0], [0,0]]" 的字符串，或者是预设的 key
        # 为了安全，建议直接使用默认值或简单的 eval
        try:
            args.medusa_choices = eval(args.medusa_choices)
        except:
            print(f"Warning: Could not eval medusa_choices string, using default.")
            args.medusa_choices = None

    # 5. 加载问题
    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    print(f"Loaded {len(questions)} questions.")

    # 准备输出文件
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    # 清空或追加模式，这里使用追加模式，如果文件不存在则创建
    
    # 6. Warmup (预热 Jittor 编译)
    print("Warming up...")
    
    # === [关键] 先同步，确保模型加载没有错误 ===
    try:
        jt.sync_all()
        print("Model loading verified OK.")
    except Exception as e:
        print(f"[FATAL] Model loading failed: {e}")
        print("Please check: 1. CUDA version compatibility, 2. Model weights integrity")
        import sys
        sys.exit(1)
    
    # === [关键] 清空 Jittor 缓存，避免之前运行的残留状态 ===
    jt.gc()
    
    dummy_input = jt.array(tokenizer(["Hello"], return_tensors="np")["input_ids"])
    
    # === [关键] warmup 时使用 try-except 捕获错误 ===
    try:
        with jt.no_grad():
            for _ in model.medusa_generate(dummy_input, max_steps=5, tokenizer=tokenizer): 
                pass
        jt.sync_all()  # 强制同步，捕获异步错误
        jt.gc()  # 清理 warmup 产生的缓存
        print("Warmup done.")
    except Exception as e:
        print(f"[FATAL] Warmup failed: {e}")
        print("This usually means a bug in Medusa tree decoding or KV cache.")
        import sys
        sys.exit(1)

    # 7. 主推理循环
    for question in tqdm(questions):
        choices = []
        
        # 处理每个 Choice (通常为 1)
        for i in range(args.num_choices):
            # 初始化对话模板
            conv = SimpleConversation()
            
            turns_output = []
            idxs_list = []      # 步数 (steps)
            new_tokens_list = [] # 实际 token 数
            wall_time_list = []  # 时间
            
            # 处理多轮对话
            for turn_idx, qs in enumerate(question["turns"]):
                # 构建 Prompt
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                input_ids = tokenizer([prompt], return_tensors="np")["input_ids"]
                
                # === [新增] 长度检查与保护 ===
                # 获取模型支持的最大长度 (Vicuna v1.3 通常是 2048，有些是 4096)
                # 安全起见，预留一部分给生成的 token
                max_model_len = 2048 
                if hasattr(model.config, "max_position_embeddings"):
                    max_model_len = model.config.max_position_embeddings
                
                curr_len = input_ids.shape[1]
                max_gen_len = args.max_new_token
                
                # 如果 (当前长度 + 生成长度) > 模型极限，进行处理
                if curr_len + max_gen_len > max_model_len:
                    # 策略1: 动态减少生成的最大长度 (推荐，保留完整 Prompt)
                    adjusted_max_new_token = max(1, max_model_len - curr_len - 16) # 留16个余量
                    print(f"Warning: QID {question['question_id']} input length {curr_len} is too long. Reducing max_new_token from {max_gen_len} to {adjusted_max_new_token}")
                    
                    # 临时修改 args 中的参数传递给推理函数
                    # 注意：这里不能修改全局 args，否则会影响后续问题，所以我们修改调用方式
                    current_max_steps = adjusted_max_new_token
                else:
                    current_max_steps = args.max_new_token

                # 如果 Prompt 本身就超长了，必须截断 Prompt (虽然这对回答质量有影响，但能防止崩机)
                if curr_len >= max_model_len:
                     print(f"Warning: QID {question['question_id']} input length {curr_len} exceeds model limit {max_model_len}. Truncating input.")
                     # 保留最后面的 max_model_len - 64 个 token
                     input_ids = input_ids[:, -(max_model_len - 64):]
                     current_max_steps = 64 # 只允许生成少量

                try:
                    # 执行推理 (使用 current_max_steps)
                    # 注意：这里我们手动修改 medusa_inference 的调用，使其接受 max_steps 参数，或者临时修改 args
                    
                    # 为了不改动 medusa_inference 的签名，我们可以临时覆盖 args.max_new_token
                    original_max_token = args.max_new_token
                    args.max_new_token = current_max_steps
                    
                    output_text, new_tokens, wall_time, steps = medusa_inference(
                        model, tokenizer, input_ids, args
                    )
                    
                    # 恢复 args
                    args.max_new_token = original_max_token
                    
                    # 记录结果
                    turns_output.append(output_text)
                    idxs_list.append(steps)
                    new_tokens_list.append(new_tokens)
                    wall_time_list.append(wall_time)
                    
                    # 更新历史用于下一轮
                    conv.messages[-1][-1] = output_text
                    
                except Exception as e:
                    print(f"Error processing Question ID {question['question_id']}: {e}")
                    # 清理 CUDA 状态，尝试恢复
                    try:
                        jt.sync_all()
                        jt.gc()
                    except:
                        pass
                    turns_output.append("ERROR")
                    idxs_list.append(0)
                    new_tokens_list.append(0)
                    wall_time_list.append(0.0)

            choices.append({
                "index": i,
                "turns": turns_output,
                "new_tokens": new_tokens_list,
                "wall_time": wall_time_list,
                "idxs": idxs_list
            })

        # 写入结果 (JSON Lines 格式)
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": get_short_uuid(),
            "model_id": args.model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        
        with open(args.answer_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 路径参数
    parser.add_argument("--base_model_path", type=str, required=True, help="HuggingFace model path (for config/tokenizer)")
    parser.add_argument("--jittor_base_weights", type=str, required=True, help="Converted Jittor base model weights (.pkl/jtr)")
    parser.add_argument("--medusa_weights", type=str, required=True, help="Trained Medusa heads weights")
    parser.add_argument("--question_file", type=str, required=True, help="Input question jsonl file")
    parser.add_argument("--answer_file", type=str, required=True, help="Output answer jsonl file")
    
    # 模型参数
    parser.add_argument("--model_id", type=str, default="medusa-jittor")
    parser.add_argument("--medusa_num_heads", type=int, default=5)
    parser.add_argument("--max_new_token", type=int, default=1024)
    parser.add_argument("--num_choices", type=int, default=1)
    
    # Medusa 采样参数
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--posterior_threshold", type=float, default=0.09)
    parser.add_argument("--posterior_alpha", type=float, default=0.3)
    parser.add_argument("--medusa_choices", type=str, default="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0]]", help="Tree structure choices")

    # 调试参数
    parser.add_argument("--question_begin", type=int, default=None)
    parser.add_argument("--question_end", type=int, default=None)

    args = parser.parse_args()
    
    run_eval(args)