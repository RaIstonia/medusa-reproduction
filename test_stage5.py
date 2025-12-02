import jittor as jt
import numpy as np
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def decode(self, ids, **kwargs):
        # 简单将 ID 转为字符串用于打印
        return " ".join([str(i) for i in ids])

def test_medusa_generation_loop():
    print("=== 开始 Phase 5 端到端推理循环测试 ===")
    
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("Using CUDA")
    
    # 1. 配置参数
    vocab_size = 200 # [MODIFIED] 稍微大一点，避免 TOPK 问题
    hidden_size = 128
    heads = 4
    
    # [FIX] 必须至少为 4，推荐 5，以匹配默认的 DEFAULT_MEDUSA_CHOICES
    medusa_heads = 5 
    
    # Configs
    llama_config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, 
        num_attention_heads=heads, num_key_value_heads=heads,
        num_hidden_layers=2, intermediate_size=256
    )
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_heads, medusa_num_layers=1,
        hidden_size=hidden_size, vocab_size=vocab_size
    )
    
    # 2. 初始化模型
    print("Initializing models...")
    base_model = LlamaForCausalLM(llama_config)
    model = MedusaModel(medusa_config, base_model=base_model)
    
    # 3. 准备输入
    # Prompt: [1, 2, 3]
    # 确保输入 ID 在 vocab_size 范围内
    input_ids = jt.array([[1, 2, 3]]).int64()
    
    tokenizer = MockTokenizer()
    
    print("Starting generation loop...")
    
    # 4. 运行生成
    gen_step = 0
    try:
        # 使用 fast=True (Greedy) 模式
        for output in model.medusa_generate(
            input_ids, 
            temperature=0.0, 
            max_steps=5, 
            tokenizer=tokenizer
        ):
            gen_step += 1
            print(f"Step {gen_step} output IDs: {output['ids']}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
        
    print(f"Successfully ran {gen_step} steps.")
    assert gen_step > 0
    print("=== Phase 5 测试通过! ===")

if __name__ == "__main__":
    test_medusa_generation_loop()