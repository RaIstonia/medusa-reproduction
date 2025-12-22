import jittor as jt
from jittor import nn
import math

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
        
        # A: 高斯初始化
        self.lora_A = nn.init.gauss(jt.zeros((r, self.in_features)), std=1./r)
        # B: 零初始化 (保证初始状态无影响)
        self.lora_B = jt.zeros((self.out_features, r))
        
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else None

    def execute(self, x):
        # 基础路径 (无梯度)
        base_out = self.base_layer(x)
        
        # LoRA 路径 (有梯度)
        # x: [batch, seq, in]
        # lora_out = (x @ A.T @ B.T) * scaling
        if self.dropout:
            x = self.dropout(x)
            
        # 显式计算以确保 Jittor 捕获梯度
        lora_out = x @ self.lora_A.transpose() @ self.lora_B.transpose()
        
        return base_out + lora_out * self.scaling

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