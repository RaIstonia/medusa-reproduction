import jittor as jt
from jittor import nn
# from transformers import PretrainedConfig
import warnings
import copy

from .kv_cache_u import initialize_past_key_values
from .utils import (
    generate_medusa_buffers, 
    initialize_medusa, 
    reset_medusa_mode, 
    generate_candidates, 
    tree_decoding, 
    evaluate_posterior, 
    update_inference_inputs
)

DEFAULT_MEDUSA_CHOICES = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], 
    [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], 
    [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], 
    [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], 
    [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], 
    [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], 
    [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]
]

class PretrainedConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
        
    def to_dict(self):
        return copy.deepcopy(self.__dict__)

class MedusaConfig(PretrainedConfig):
    """
    Medusa 配置类，存储 Medusa Head 的超参数。
    """
    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    Medusa Head 使用的残差块。
    包含一个 Linear 层和一个 SiLU 激活函数。
    """
    def __init__(self, hidden_size):
        super().__init__()
        # 注意：Jittor 的 Linear 默认初始化分布可能不同
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        
        # 初始化策略：为了保证初始状态下 Medusa Head 不影响原模型或输出接近 0
        # 将权重初始化为 0 (类似原版逻辑)
        jt.init.constant_(self.linear.weight, 0.0)
        if self.linear.bias is not None:
            jt.init.constant_(self.linear.bias, 0.0)

    def execute(self, x):
        """
        Args:
            x (jt.Var): Input tensor [batch, seq_len, hidden_size]
        """
        return x + self.act(self.linear(x))

class MedusaModel(nn.Module):
    """
    Medusa 模型主体。
    它包装了一个基础的大语言模型 (base_model)，并附加了多个 Medusa Heads。
    """
    def __init__(self, config, base_model=None):
        """
        Args:
            config (MedusaConfig): 配置对象
            base_model (nn.Module, optional): 预先加载好的基础模型实例 (如 LlamaForCausalLM 的 Jittor 版本)
        """
        super().__init__()
        self.config = config
        self.medusa_num_heads = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers
        self.base_model_name_or_path = config.base_model_name_or_path
        
        # 基础模型参数
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # 保存 Base Model
        # 如果初始化时未传入，后续可以通过 set_base_model 方法设置
        self.base_model = base_model

        # 创建 Medusa Heads
        # 结构：List[ Sequential( ResBlock * k, Linear ) ]
        self.medusa_head = nn.ModuleList([
            nn.Sequential(
                *([ResBlock(self.hidden_size)] * self.medusa_num_layers),
                nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            )
            for _ in range(self.medusa_num_heads)
        ])
        
        # 缓存 buffer 占位符 (将在 generate 阶段初始化)
        self.medusa_buffers = None
        self.medusa_choices = None

    def set_base_model(self, base_model):
        """后期设置基础模型"""
        self.base_model = base_model

    def get_tokenizer(self):
        """获取 tokenizer (通常由外部管理，这里保留接口兼容性)"""
        if hasattr(self, 'tokenizer'):
            return self.tokenizer
        # 如果没有，尝试从 base_model 获取
        if self.base_model and hasattr(self.base_model, 'tokenizer'):
            return self.base_model.tokenizer
        raise AttributeError("Tokenizer not found in MedusaModel")

    def execute(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """
        前向传播逻辑。
        
        Args:
            medusa_forward (bool): 如果为 True，则计算 Medusa Heads 的输出。
                                   如果为 False，仅运行 base_model (用于 prefill 或普通生成)。
        """
        # 1. 如果不是 Medusa 模式，直接透传给 Base Model
        if not medusa_forward:
            if self.base_model is None:
                raise RuntimeError("Base model is not initialized.")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                return_dict=True, # 强制返回对象以便后续处理
                **kwargs
            )

        # 2. Medusa 模式 (通常用于 Medusa 头的训练或树形验证步骤)
        # Jittor 默认不计算梯度，训练时需外部控制 start_grad
        
        with jt.no_grad(): # 在 Medusa 推理模式下，Base Model 不需要计算梯度
             base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=True, # 关键：必须请求 hidden_states
                return_dict=True,          # 关键：强制返回字典对象
                **kwargs,
            )
        
        # 解析 Base Model 输出
        # base_outputs.hidden_states 是一个元组，包含每一层的输出
        # 最后一个元素 ([-1]) 是经过了 LlamaRMSNorm 后的最终隐状态，正是 Medusa Head 的输入
        last_hidden_state = base_outputs.hidden_states[-1]

        hs = last_hidden_state.clone().float32()

        # 2. [关键] Sync: 强制 Jittor 立即执行 Base Model 的计算
        # 这会将惰性的计算图实体化为真实数据
        hs.sync()
        
        # 3. [关键] Stop Grad & Clone: 彻底切断与 Base Model 计算图的联系
        # 此时 hs 变成了一个纯粹的数据节点，没有任何历史包袱
        hs = hs.stop_grad().clone()
        
        # 4. [关键] 删除引用并强制 GC:
        # 手动删除 base_outputs，告诉 Jittor "Base Model 的中间结果我不要了"
        del base_outputs
        del last_hidden_state
        # 强制触发 Jittor 的垃圾回收，立即释放 Base Model 前向传播产生的临时显存
        jt.gc()

        medusa_logits = []
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i](hs))
            
        # 堆叠结果: [num_heads, batch, seq_len, vocab_size]
        medusa_logits_stack = jt.stack(medusa_logits, dim=0)

        if output_orig:
            # LlamaForCausalLM 已经计算了 logits，直接复用
            # orig_logits = base_outputs.logits
            # return medusa_logits_stack, base_outputs, orig_logits
            raise NotImplementedError("Training with output_orig=True requires modification to memory optimization logic.")
            
        return medusa_logits_stack

    def load_medusa_weights(self, weight_path):
        """
        专门加载 Medusa Head 的权重。
        """
        print(f"Loading Medusa heads from {weight_path}")
        state_dict = jt.load(weight_path)
        self.medusa_head.load_parameters(state_dict)

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        medusa_choices=None,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True,
        tokenizer = None
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        if medusa_choices is None:
            medusa_choices = DEFAULT_MEDUSA_CHOICES

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            medusa_buffers = self.medusa_buffers
        else:
            medusa_buffers = generate_medusa_buffers(medusa_choices, device="cuda")
            self.medusa_buffers = medusa_buffers
            self.medusa_choices = medusa_choices

        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model)
        
        reset_medusa_mode(self)

        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        input_len = input_ids.shape[1]

        for idx in range(max_steps):
            # (A) Generate Candidates
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # (B) Tree Decoding
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # (C) Evaluate Posterior
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # (D) Update
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            # 测试脚本需要: step, accept_length, 和 ids
            # 获取所有新生成的 ids (转为 list)
            curr_ids = input_ids[0, input_len:].numpy().tolist()
            
            if tokenizer is not None:
                text = tokenizer.decode(curr_ids, skip_special_tokens=True)
                
                # yield 一个包含丰富信息的字典
                yield {
                    "text": text,
                    "ids": curr_ids,          # 测试脚本需要统计 len(ids)
                    "accept_length": accept_length, # 用于统计加速效率
                    "step": idx               # 当前步数 (idxs)
                }
                
                # Check EOS
                if tokenizer.eos_token_id in curr_ids:
                    break
            else:
                # 即使没有 tokenizer，也返回结构化数据
                yield {
                    "ids": curr_ids,
                    "accept_length": accept_length,
                    "step": idx
                }