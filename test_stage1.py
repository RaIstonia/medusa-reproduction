import jittor as jt
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel, ResBlock

def test_phase1():
    print("=== 开始 Phase 1 测试 ===")
    
    # 1. 测试 Config
    config = MedusaConfig(
        medusa_num_heads=3,
        medusa_num_layers=1,
        hidden_size=64,   # 使用小尺寸进行测试
        vocab_size=100
    )
    print(f"[Pass] Config created: {config.medusa_num_heads} heads")

    # 2. 测试 ResBlock
    block = ResBlock(hidden_size=64)
    x = jt.randn(1, 10, 64)
    y = block(x)
    assert y.shape == [1, 10, 64]
    print(f"[Pass] ResBlock forward shape: {y.shape}")

    # 3. 测试 MedusaModel 初始化 (无 Base Model)
    model = MedusaModel(config)
    print(f"[Pass] MedusaModel initialized with {len(model.medusa_head)} heads")

    # 4. 检查参数量和结构
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total params in Medusa Heads: {total_params}")
    
    # 验证 Head 结构
    head_0 = model.medusa_head[0]
    # 应该包含 1个 ResBlock 和 1个 Linear
    print(f"Head 0 structure: {head_0}")

    # 5. 模拟一次 Head 的前向传播
    # 假设我们拿到了 hidden_states
    hidden_states = jt.randn(1, 5, 64)
    medusa_logits = []
    for head in model.medusa_head:
        medusa_logits.append(head(hidden_states))
    
    stacked = jt.stack(medusa_logits, dim=0)
    assert stacked.shape == [3, 1, 5, 100] # [heads, batch, seq, vocab]
    print(f"[Pass] Medusa Heads manual forward shape: {stacked.shape}")

    print("=== Phase 1 测试全部通过 ===")

if __name__ == "__main__":
    # 开启 GPU (如果有)
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    test_phase1()