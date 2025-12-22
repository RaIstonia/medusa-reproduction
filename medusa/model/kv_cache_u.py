import jittor as jt

class KVCache:
    """
    A key-value cache for the model (Jittor Version).
    
    注意：Jittor 中对数组元素 slice 调用 assign() 不会更新原数组，
    必须使用 setitem (parent_array[idx] = value) 才能正确更新。
    因此这个类保存对父数组的引用和索引，而不是保存标量视图。
    """

    def __init__(self, data, current_length_array, length_idx):
        """
        Args:
            data (jt.Var): The data tensor for this cache. [Batch, Heads, MaxLen, Dim]
            current_length_array (jt.Var): The shared length array. [Total_Layers * 2]
            length_idx (int): The index into current_length_array for this cache.
        """
        self.data = data
        self._length_array = current_length_array  # 保存对父数组的引用
        self._length_idx = length_idx  # 保存索引
    
    @property
    def current_length(self):
        """返回当前长度（标量 Var）"""
        return self._length_array[self._length_idx]
    
    def _set_current_length(self, value):
        """正确更新父数组中的长度值"""
        self._length_array[self._length_idx] = value

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
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
        if dim == 2:
            tgt = self.data[:, :, indices, :]
        else:
            indices_list = [slice(None)] * len(self.data.shape)
            indices_list[dim] = indices
            tgt = self.data[tuple(indices_list)]

        tgt_len = tgt.shape[dim]
        
        slices = [slice(None)] * len(self.data.shape)
        slices[dim] = slice(prev_length, prev_length + tgt_len)
        
        self.data[tuple(slices)] = tgt
        
        # 使用正确的方式更新长度
        self._set_current_length(prev_length + tgt_len)

    def cat(self, tensor: jt.Var, dim: int = 2):
        """
        Concatenate the given tensor with the current data.
        通常用于 Decoding 阶段将新生成的 Token KV 写入 Cache。
        """
        # 获取当前写入位置
        start_idx = self.current_length.item()
        add_len = tensor.shape[dim]
        end_idx = start_idx + add_len
        
        # 写入数据 (In-place update)
        slices = [slice(None)] * len(self.data.shape)
        slices[dim] = slice(start_idx, end_idx)
        self.data[tuple(slices)] = tensor
        
        # 使用正确的方式更新长度
        self._set_current_length(end_idx)
        
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
    
    dtype = model.dtype if hasattr(model, 'dtype') else "float16"

    # 1. 预分配显存
    # Shape: [Total_Layers * 2, Batch, Heads, Max_Len, Head_Dim]
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
    # Shape: [Total_Layers * 2]
    current_length_data = jt.zeros(
        (config.num_hidden_layers * 2,), 
        dtype="int32"
    )

    # 3. 创建 KVCache 对象列表
    # 注意：传入父数组引用和索引，而不是标量视图
    past_key_values = [] 
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(
                    past_key_values_data[i * 2 + j], 
                    current_length_data,  # 传入整个数组
                    i * 2 + j             # 传入索引
                )
                for j in range(2)  # K and V
            ]
        )
        
    return past_key_values, past_key_values_data, current_length_data
