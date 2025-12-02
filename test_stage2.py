import jittor as jt
import numpy as np
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model.modeling_llama import LlamaForCausalLM
from medusa.model.llama_src.llama_config import LlamaConfig

def test_integration():
    print("=== 开始 Medusa + Llama 集成测试 ===")
    
    # 1. 配置参数 (使用极小配置以加快速度)
    vocab_size = 1000
    hidden_size = 64
    num_heads = 4 # Base model heads
    medusa_heads = 3
    seq_len = 10
    batch_size = 2
    
    # Base Llama Config
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        pad_token_id=0
    )
    
    # Medusa Config
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_heads,
        medusa_num_layers=1,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        base_model_name_or_path="test_path"
    )

    print("[1] 初始化模型...")
    # 初始化 Base Model (LlamaForCausalLM)
    base_model = LlamaForCausalLM(llama_config)
    
    # 初始化 Medusa Model 并挂载 Base Model
    model = MedusaModel(medusa_config, base_model=base_model)
    print("模型初始化完成。")

    # 2. 构造虚拟输入
    input_ids = jt.randint(0, vocab_size, shape=(batch_size, seq_len))
    attention_mask = jt.ones((batch_size, seq_len))

    # 3. 测试场景 A: 普通 Forward (Medusa_forward=False)
    print("\n[2] 测试普通 Forward (Base Model Pass)...")
    outputs_base = model.execute(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        medusa_forward=False
    )
    # 检查是否返回了 CausalLMOutputWithPast 对象
    assert hasattr(outputs_base, 'logits'), "Base outputs 应该包含 logits"
    print(f"Base Logits shape: {outputs_base.logits.shape}") # Should be [B, S, V]
    assert outputs_base.logits.shape == [batch_size, seq_len, vocab_size]

    # 4. 测试场景 B: Medusa Forward (Only Heads)
    print("\n[3] 测试 Medusa Forward (Heads Only)...")
    medusa_logits = model.execute(
        input_ids=input_ids,
        attention_mask=attention_mask,
        medusa_forward=True,
        output_orig=False
    )
    # 预期形状: [medusa_heads, batch, seq_len, vocab_size]
    expected_shape = [medusa_heads, batch_size, seq_len, vocab_size]
    print(f"Medusa Logits shape: {medusa_logits.shape}")
    assert medusa_logits.shape == expected_shape, f"期望 {expected_shape}, 实际 {medusa_logits.shape}"

    # 5. 测试场景 C: Medusa Forward + Output Original
    print("\n[4] 测试 Medusa Forward + Output Orig...")
    medusa_logits_2, base_out, orig_logits = model.execute(
        input_ids=input_ids,
        attention_mask=attention_mask,
        medusa_forward=True,
        output_orig=True
    )
    
    print(f"Original Logits shape: {orig_logits.shape}")
    assert orig_logits.shape == [batch_size, seq_len, vocab_size]
    # 验证两次 Medusa 输出是否一致
    diff = (medusa_logits - medusa_logits_2).abs().sum().item()
    print(f"Diff between runs: {diff}")
    assert diff < 1e-5, "两次运行的 Medusa 输出应该一致"

    print("\n=== 集成测试通过! ===")

if __name__ == "__main__":
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("Using CUDA.")
    else:
        print("Using CPU.")
    test_integration()