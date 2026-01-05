import jittor as jt
from jittor import nn
import math
import numpy as np

class LoRAConfig:
    def __init__(self, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r, lora_alpha, lora_dropout):
        super().__init__()
        # 1. 冻结并保存原始层
        self.base_layer = original_layer
        for param in self.base_layer.parameters():
            param.stop_grad()
            
        # 2. LoRA 参数
        self.r = r
        self.scaling = lora_alpha / r
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # --- [关键修复 1] 优化初始化 ---
        # 原代码: std=1./r (0.0625) 太大了，导致初始 loss 爆炸
        # 改为更小的 std，例如 0.01，让 LoRA 初始权重更小，接近于 0
        # 这样可以避免破坏原模型的分布，Base Loss 应该会降到 2-3 左右
        init_std = 0.01
        # 使用 numpy 生成随机数，然后转换为 Jittor 数组
        lora_A_np = np.random.normal(0.0, init_std, size=(r, self.in_features)).astype(np.float32)
        self.lora_A = jt.array(lora_A_np)
        
        # B 保持全 0 初始化，保证初始状态下 LoRA 输出为 0，不干扰原模型
        self.lora_B = jt.zeros((self.out_features, r)).float32()

        self.dropout = nn.Dropout(p=lora_dropout) 

    def execute(self, x):
        # 1. 基础路径
        base_out = self.base_layer(x)
        
        # 2. LoRA 路径 (强制 float32 计算)
        # input_dtype 保存原始精度 (如 float16)
        input_dtype = x.dtype
        
        # 转为 FP32 进行计算，避免中间结果溢出
        x_lora = x.float32()
        x_lora = self.dropout(x_lora)
            
        # 计算: x @ A^T @ B^T
        # 此时全链路都是 float32
        lora_out = x_lora @ self.lora_A.transpose() @ self.lora_B.transpose()
        
        # --- [关键修复 2] 缩放时的稳定性 ---
        # 确保 scaling 是 float 标量
        lora_out = lora_out * self.scaling
        
        if input_dtype == jt.float16:
            # 尝试在 FP32 相加，防止求和溢出
            return (base_out.float32() + lora_out).astype(input_dtype)
        else:
            return base_out + lora_out.astype(input_dtype)

def inject_lora(model, config: LoRAConfig):
    """
    遍历模型，将指定的 Linear 层替换为 LoRALinear
    """
    target_modules = config.target_modules
    replaced_count = 0
    
    # 收集需要替换的层名称
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查名称后缀是否在目标列表中
            if any(name.endswith(t) for t in target_modules):
                modules_to_replace.append(name)
    
    # 执行替换
    for name in modules_to_replace:
        # 获取父模块和子模块名称
        path = name.split('.')
        parent = model
        for p in path[:-1]:
            parent = getattr(parent, p)
        target_name = path[-1]
        original_module = getattr(parent, target_name)
        
        # 创建 Wrapper
        lora_wrapper = LoRALinear(
            original_module, 
            r=config.r, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout
        )
        
        # 替换
        setattr(parent, target_name, lora_wrapper)
        replaced_count += 1
        
    print(f"LoRA Injection applied: replaced {replaced_count} layers.")
    return model

def mark_only_lora_as_trainable(model):
    """
    辅助函数：标记梯度
    1. Base Model 原参数 -> stop_grad
    2. LoRA 参数 -> start_grad
    3. Medusa Head -> start_grad (在外部处理)
    """
    # 遍历所有参数
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.start_grad()
        else:
            param.stop_grad()