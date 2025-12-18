import argparse
import json
import os
import time
import uuid
import jittor as jt
import numpy as np
from tqdm import tqdm

# 依赖 HuggingFace transformers 进行 tokenizer 处理
from transformers import AutoTokenizer

# 导入你的模型定义
# 假设你的文件结构是 medusa/model/modeling_medusa.py 和 medusa/model/modeling_llama.py
from medusa.model.modeling_medusa import MedusaModel, MedusaConfig
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig

# 开启 CUDA
jt.flags.use_cuda = 1

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
        for role, content in self.messages:
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
    # Jittor 需要确保 input_ids 是 jt.Var
    if not isinstance(input_ids, jt.Var):
        input_ids = jt.array(input_ids)
    
    input_len = input_ids.shape[1]
    
    # 同步以确保计时准确
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
            max_steps=args.max_new_token,
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
    dummy_input = jt.array(tokenizer(["Hello"], return_tensors="np")["input_ids"])
    with jt.no_grad():
        for _ in model.medusa_generate(dummy_input, max_steps=5, tokenizer=tokenizer): pass
    jt.sync_all()
    print("Warmup done.")

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
                
                try:
                    # 执行推理
                    output_text, new_tokens, wall_time, steps = medusa_inference(
                        model, tokenizer, input_ids, args
                    )
                    
                    # 记录结果
                    turns_output.append(output_text)
                    idxs_list.append(steps)
                    new_tokens_list.append(new_tokens)
                    wall_time_list.append(wall_time)
                    
                    # 更新历史用于下一轮
                    conv.messages[-1][-1] = output_text
                    
                except Exception as e:
                    print(f"Error processing Question ID {question['question_id']}: {e}")
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