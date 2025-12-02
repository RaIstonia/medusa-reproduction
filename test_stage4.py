import jittor as jt
import numpy as np
from medusa.model.utils import generate_medusa_buffers

# Mock Choice Data
# 排序后的顺序应该是:
# Index 0: Root
# Index 1: [0]
# Index 2: [1]
# Index 3: [0, 0] (Child of 1)
# Index 4: [0, 1] (Child of 1)
# Index 5: [1, 0] (Child of 2)
mock_choices = [
    [0],
    [0, 0],
    [1],
    [0, 1],
    [1, 0],
]

def test_buffer_generation():
    print("=== 开始 Phase 3 Buffer 生成测试 (Corrected) ===")
    
    # 1. 生成 Buffer
    buffers = generate_medusa_buffers(mock_choices, device="cpu")
    
    print("Buffers generated:")
    for k, v in buffers.items():
        # 打印形状和类型，方便调试
        print(f" - {k}: {v.shape}, dtype={v.dtype}")
        
    # 2. 验证 Attention Mask
    # Mask 应该是 [1, 1, 6, 6]
    mask = buffers["medusa_attn_mask"].numpy()
    assert mask.shape == (1, 1, 6, 6)
    
    # --- 修正后的断言 ---
    
    # Case A: 验证 Index 3 ([0, 0]) 是否依赖 Index 1 ([0])
    # 这是一条正常的父子路径
    val_3_1 = mask[0,0,3,1]
    print(f"Mask check (3->1, [0,0]->[0]): {val_3_1}")
    assert val_3_1 == 1.0, f"Index 3 should depend on Index 1, got {val_3_1}"
    
    # Case B: 验证 Index 5 ([1, 0]) 是否依赖 Index 2 ([1])
    val_5_2 = mask[0,0,5,2]
    print(f"Mask check (5->2, [1,0]->[1]): {val_5_2}")
    assert val_5_2 == 1.0, f"Index 5 should depend on Index 2, got {val_5_2}"
    
    # Case C: 验证 Index 2 ([1]) 与 Index 1 ([0]) 是否独立
    # 它们是兄弟节点，不应有 Attention
    val_2_1 = mask[0,0,2,1]
    print(f"Mask check (2->1, [1]->[0]): {val_2_1}")
    assert val_2_1 == 0.0, f"Index 2 should NOT depend on Index 1, got {val_2_1}"
    
    # 3. 验证 Tree Indices
    # tree_indices[i] 存储的是 token 在 vocab 中的偏移量 或者 相对 ID
    ti = buffers["tree_indices"].numpy()
    print(f"Tree Indices: {ti}")
    assert len(ti) == 6
    
    # 4. 验证 Retrieve Indices
    ri = buffers["retrieve_indices"].numpy()
    print(f"Retrieve Indices shape: {ri.shape}")
    # 根据我们推导的逻辑，应该是 3 条路径 (因为 [1] 和 [0] 被包含在更长路径中了)
    # Shape 应该是 (3, 3) -> [0, parent, child]
    assert ri.shape == (3, 3) 
    
    print("=== Phase 3 Buffer 测试通过! ===")

if __name__ == "__main__":
    test_buffer_generation()