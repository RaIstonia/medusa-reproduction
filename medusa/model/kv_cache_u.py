import jittor as jt

class KVCache:
    """
    A key-value cache for the model (Jittor Version).
    """

    def __init__(self, data, current_length):
        """
        Args:
            data (jt.Var): Initial tensor to store the keys and values. [Batch, Heads, MaxLen, Dim]
            current_length (jt.Var): Initial length of the data. (Scalar)
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        # 注意：这里需要 .item() 将 Jittor Var 转为 Python int，否则不能作为形状返回
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: jt.Var, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.
        用于 Tree Attention 验证阶段，将选中的 Token KV 复制到连续位置。
        """
        # PyTorch: tgt = self.data.index_select(dim, indices)
        # Jittor: 使用高级索引。假设 dim=2 (seq_len维度)
        # self.data shape: [batch, heads, seq_len, head_dim]
        if dim == 2:
            tgt = self.data[:, :, indices, :]
        else:
            # 通用实现，但通常 KV Cache 都在 dim 2 操作
            indices_list = [slice(None)] * len(self.data.shape)
            indices_list[dim] = indices
            tgt = self.data[tuple(indices_list)]

        # PyTorch: dst.copy_(tgt)
        # Jittor: 原位赋值
        tgt_len = tgt.shape[dim]
        
        # 构造切片: data[:, :, prev_length : prev_length+len, :] = tgt
        slices = [slice(None)] * len(self.data.shape)
        slices[dim] = slice(prev_length, prev_length + tgt_len)
        
        self.data[tuple(slices)] = tgt
        
        # 更新长度
        # self.current_length.fill_(prev_length + tgt_len) -> Jittor assign
        self.current_length.assign(prev_length + tgt_len)

    def cat(self, tensor: jt.Var, dim: int = 2):
        """
        Concatenate the given tensor with the current data.
        通常用于 Decoding 阶段将新生成的 Token KV 写入 Cache。
        """
        # 获取当前写入位置 (必须转为 int 用于切片)
        start_idx = self.current_length.item()
        add_len = tensor.shape[dim]
        end_idx = start_idx + add_len
        
        # 写入数据 (In-place update)
        # data[:, :, start:end, :] = tensor
        slices = [slice(None)] * len(self.data.shape)
        slices[dim] = slice(start_idx, end_idx)
        self.data[tuple(slices)] = tensor
        
        # 更新长度指针
        self.current_length.assign(end_idx)
        
        # 返回当前的有效数据视图 (用于 Attention 计算)
        return_slices = [slice(None)] * len(self.data.shape)
        return_slices[dim] = slice(0, end_idx)
        
        return self.data[tuple(return_slices)]


def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.
    """
    config = model.config
    batch_size = 1
    
    # 获取数据类型，如果是字符串则保持，如果是对象则取属性
    dtype = model.dtype if hasattr(model, 'dtype') else "float16"

    # 1. 预分配显存 (Jittor Lazy Execution 会在第一次使用时分配)
    # Shape: [Total_Layers * 2, Batch, Heads, Max_Len, Head_Dim]
    # Total_Layers * 2 是因为每层有 K 和 V 两个 Cache
    past_key_values_data = jt.zeros(
        (
            config.num_hidden_layers * 2,
            batch_size,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
        ),
        dtype=dtype
    )
    
    # 2. 长度计数器
    # 放在 CPU 还是 GPU 在 Jittor 中由框架管理，这里初始化为 int32 即可
    # Shape: [Total_Layers * 2]
    current_length_data = jt.zeros(
        (config.num_hidden_layers * 2,), 
        dtype="int32"
    )

    # 3. 创建 KVCache 对象列表
    past_key_values = [] 
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2) # K and V
            ]
        )
        
    return past_key_values, past_key_values_data, current_length_data