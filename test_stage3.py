import jittor as jt
import numpy as np
from medusa.model.kv_cache_u import KVCache, initialize_past_key_values

# 模拟一个简单的 Config 对象
class MockConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.num_key_value_heads = 4
        self.max_position_embeddings = 1024
        self.hidden_size = 128 # 4 heads * 32 dim
        self.num_attention_heads = 4

class MockModel:
    def __init__(self):
        self.config = MockConfig()
        self.dtype = "float32"
        self.device = "cuda" # Jittor 自动处理，但也兼容参数

def test_kv_cache_logic():
    print("=== 开始 Phase 2 KV Cache 逻辑测试 ===")
    
    model = MockModel()
    
    # 1. 测试初始化
    print("[1] 初始化 KV Cache...")
    past_key_values, raw_data, len_data = initialize_past_key_values(model)
    
    # 检查结构
    assert len(past_key_values) == model.config.num_hidden_layers
    assert len(past_key_values[0]) == 2 # (K, V)
    
    # 检查底层 Tensor 共享
    # 修改 KVCache 应该反映在 raw_data 上
    cache_k0 = past_key_values[0][0]
    print(f"Initial length: {cache_k0.current_length.item()}")
    assert cache_k0.current_length.item() == 0

    # 2. 测试 cat (写入新 Token)
    print("\n[2] 测试 cat 操作 (Append)...")
    # 模拟输入: [Batch=1, Heads=4, Seq=1, Dim=32]
    new_k = jt.ones((1, 4, 1, 32)) * 1.5 
    
    # 执行 cat
    valid_data = cache_k0.cat(new_k)
    
    # 验证长度更新
    current_len = cache_k0.current_length.item()
    print(f"Length after cat 1 token: {current_len}")
    assert current_len == 1
    
    # 验证数据正确性
    assert valid_data.shape == [1, 4, 1, 32]
    assert valid_data[0,0,0,0].item() == 1.5
    
    # 验证 raw_data 是否被更新 (关键：确保是在预分配内存上操作)
    # raw_data 的第0个切片对应 layer0 的 K
    assert raw_data[0, 0, 0, 0, 0].item() == 1.5
    
    # 再写入 2 个 Token
    new_k2 = jt.ones((1, 4, 2, 32)) * 2.5
    cache_k0.cat(new_k2)
    print(f"Length after cat 2 tokens: {cache_k0.current_length.item()}")
    assert cache_k0.current_length.item() == 3

    # 3. 测试 copy (Tree Attention 回滚/重排)
    print("\n[3] 测试 copy 操作 (Reorder)...")
    # 假设我们现在 cache 里有 3 个 token: [1.5, 2.5, 2.5]
    # 我们想把第 0 个和 第 2 个 复制到新的位置 (模拟选中了特定的树路径)
    
    indices = jt.array([0, 2]) # 选中第1个和第3个
    prev_len = 3 # 假设接着写
    
    # 注意：copy 方法在原代码中逻辑是：把 indices 指向的数据，复制到 prev_len 开始的位置
    # 并把长度设置为 prev_len + indices.len
    # 但原代码逻辑似乎是覆盖写？让我们验证 medusa 的用法。
    # 通常 copy 是用来 "Accept Candidate" 后整理 Cache 的。
    
    # 这里我们简单测试：把 index 0 (1.5) 复制到 index 0 的位置 (覆盖)
    # 把 index 2 (2.5) 复制到 index 1 的位置
    # 目标状态: [1.5, 2.5] (长度变为2)
    
    # Medusa 的 copy 签名: copy(indices, prev_length)
    # 这里的 prev_length 实际上是 "Destination Start Index"
    
    # 场景：我们接受了 index=[0, 2] 的 token，想把它们作为新的历史
    # 所以我们把 data[indices] 复制到 data[0:]
    cache_k0.copy(indices, prev_length=0)
    
    # 验证长度
    print(f"Length after copy: {cache_k0.current_length.item()}")
    assert cache_k0.current_length.item() == 2 # 0 + 2
    
    # 验证数据
    # 第 0 个应该是原来的 1.5
    val0 = cache_k0.data[0,0,0,0].item()
    # 第 1 个应该是原来的 2.5 (来自 index 2)
    val1 = cache_k0.data[0,0,1,0].item()
    
    print(f"Value at pos 0: {val0} (Expected 1.5)")
    print(f"Value at pos 1: {val1} (Expected 2.5)")
    
    assert val0 == 1.5
    assert val1 == 2.5
    
    print("\n=== Phase 2 KV Cache 测试通过! ===")

if __name__ == "__main__":
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    test_kv_cache_logic()