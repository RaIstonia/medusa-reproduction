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
        enable_lora_training=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.enable_lora_training = enable_lora_training

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
        self.enable_lora_training = getattr(config, 'enable_lora_training', False)
        
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
        # 根据 enable_lora_training 参数决定是否保留梯度连接
        
        # 如果启用 LoRA 训练，需要保留 Base Model 的梯度计算图
        if self.enable_lora_training:
            # LoRA 训练模式：保留梯度连接，允许梯度回传到 Base Model 的 LoRA 层
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
            last_hidden_state = base_outputs.hidden_states[-1]
            hs = last_hidden_state.clone().float32()
            
            # 在训练时，保留计算图，允许梯度回传
            # 不调用 stop_grad()，保持梯度连接
            # 注意：仍然需要 sync 来确保计算完成
            hs.sync()
            
            # 如果 output_orig=True，需要保存 logits 和 base_outputs
            if output_orig:
                orig_logits = base_outputs.logits
                saved_base_outputs = base_outputs
                # 注意：在训练模式下，不删除 base_outputs，保留梯度连接
            else:
                # 清理引用（但保留梯度连接）
                del base_outputs
                del last_hidden_state
            
        else:
            # 原始模式：显存优化，切断梯度连接（仅训练 Medusa Head）
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
            
            # 如果 output_orig=True，需要保存 logits 和 base_outputs
            if output_orig:
                orig_logits = base_outputs.logits
                saved_base_outputs = base_outputs
                # 即使 output_orig=True，也需要处理 hs 以便后续使用（但不删除 base_outputs）
                hs = hs.stop_grad().clone()
            else:
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
            return medusa_logits_stack, saved_base_outputs, orig_logits
            
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
        
        # === [核心修复] 获取模型允许的最大长度和 Medusa 安全余量 ===
        # Vicuna v1.3 config usually has 2048. Safe guard with 2048 if None.
        # 从 base_model 的 config 中获取，因为 MedusaConfig 可能没有这个属性
        if self.base_model and hasattr(self.base_model, 'config'):
            max_model_len = getattr(self.base_model.config, "max_position_embeddings", 2048)
        else:
            max_model_len = 2048  # 默认值
        # 获取 Medusa 树的最大深度 (投机步长)
        # 简单估算：medusa_choices 的最大长度，或者给一个保守的 Buffer
        # 为了安全，这里给一个保守的 Buffer，比如 64
        if medusa_choices is not None:
            # 计算 medusa_choices 中的最大深度
            max_tree_depth = max(len(choice) for choice in medusa_choices) if medusa_choices else 0
            safe_margin = max(64, max_tree_depth * 2)  # 至少 64，或者树深度的 2 倍
        else:
            safe_margin = 64

        for idx in range(max_steps):
            # === [核心修复] 边界检查 ===
            # 获取当前序列长度
            current_seq_len = input_ids.shape[1]
            
            # 如果 (当前长度 + 安全余量) 超过了模型最大长度，强制停止
            # 这是为了防止 Medusa 的树形探测访问到 KV Cache 2048 之外的地方
            if current_seq_len + safe_margin >= max_model_len:
                # 触发停止条件
                if tokenizer is not None:
                    # decode 逻辑
                    curr_ids = input_ids[0, input_len:].numpy().tolist()
                    text = tokenizer.decode(curr_ids, skip_special_tokens=True)
                    yield {
                        "text": text,
                        "ids": curr_ids,
                        "accept_length": 0,
                        "step": idx
                    }
                else:
                    # 即使没有 tokenizer，也返回结构化数据
                    curr_ids = input_ids[0, input_len:].numpy().tolist()
                    yield {
                        "ids": curr_ids,
                        "accept_length": 0,
                        "step": idx
                    }
                # 强制停止，不再进行下一步的树形探测
                break
            # ==========================
            
            # (A) Generate Candidates
            # suppress_special_tokens=True: 抑制 BOS/EOS 等特殊 token
            # 避免 base model 在某些情况下给特殊 token 很高的概率导致生成跑偏
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
                suppress_special_tokens=True,
            )

            # (B) Tree Decoding
            # 保存 Tree Decoding 之前的 KV Cache 长度（用于后续回滚）
            prev_len_before_tree = input_ids.shape[1]
            
            # 确保 medusa_mask 被设置（tree_decoding 需要它）
            # 注意：在第一次 prefill 后，medusa_mask 已经被 initialize_medusa 设置了
            # 但在回滚后的 Forward 中，我们清除了它，所以这里需要重新设置
            if self.base_model.model.medusa_mask is None:
                self.base_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"]
            
            tree_medusa_logits, tree_logits, tree_outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # (C) Evaluate Posterior
            best_candidate, accept_length = evaluate_posterior(
                tree_logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # (D) Update - 重构版：指针回滚 + 标准 Forward
            # 核心思想：
            # 1. Tree Decoding 产生的 KV 是基于树状 Position ID 的，是"脏"数据，只用于验证
            # 2. 验证完成后，我们丢弃这些脏 KV，回滚指针
            # 3. 用接受的 token 重新运行标准 Forward，生成正确的线性 KV Cache
            
            # 1. 解析被接受的 Tokens
            bc_idx = best_candidate.item()
            al_val = int(accept_length)
            # accepted_tokens 包含: [t_0 (原始logits预测的), t_1, t_2... (Medusa预测并验证通过的)]
            # candidates shape: [Num_Cands, Max_Len]
            accepted_tokens = candidates[bc_idx:bc_idx+1, :al_val + 1]  # [1, K+1]
            
            # 2. 更新 input_ids
            input_ids = jt.concat([input_ids, accepted_tokens], dim=-1)
            
            # 3. 修正 KV Cache 指针（回滚到 Tree Decoding 之前）
            # Tree Decoding 在 past_key_values 里写了一堆基于树状 Position ID 的 KV，
            # 这些 KV 的位置信息是错误的，不能直接使用。
            # 我们需要把指针"拨回来"，拨到 Tree Decoding 之前的状态。
            # 这样 Tree Decoding 写入的脏数据在逻辑上被"删除"了（会被后续的正确数据覆盖）。
            for i in range(current_length_data.shape[0]):
                current_length_data[i] = prev_len_before_tree
            
            # 4. 增量 Forward (Correction Step)
            # 将 accepted_tokens 再次输入模型，使用标准的线性 Position ID。
            # 这次 Forward 会：
            # a. 用正确的线性 Position ID 计算 accepted_tokens 的 KV，并写入 Cache（覆盖掉 Tree Decoding 的脏数据）
            # b. 计算最后一个 Token 的 Logits，用于下一轮生成
            # c. 同时计算 Medusa Head 的 logits
            
            # 临时清除 medusa_mask，确保走标准 Causal Mask
            old_mask = self.base_model.model.medusa_mask
            self.base_model.model.medusa_mask = None
            
            # 执行标准 Forward（medusa_forward=True 以计算 Medusa Heads）
            # 注意：这里传入的 past_key_values 的指针已经被回滚到 prev_len_before_tree
            # 所以 Forward 会从 prev_len_before_tree 位置开始写入新的 KV
            correction_medusa_logits, correction_outputs, correction_logits = self(
                input_ids=accepted_tokens,
                past_key_values=past_key_values,
                output_orig=True,
                medusa_forward=True  # 开启以计算 Medusa Heads
            )
            
            # 恢复 mask（虽然下一轮 initialize_medusa 会重新设置，但保持状态一致）
            self.base_model.model.medusa_mask = old_mask
            
            # 5. 更新 logits 和 medusa_logits（用于下一轮 generate_candidates）
            # correction_logits shape: [1, K+1, Vocab]
            # 我们只需要最后一个位置的 logits: [1, 1, Vocab]
            logits = correction_logits[:, -1:, :]  # [1, 1, Vocab]
            
            # correction_medusa_logits shape: [Heads, 1, K+1, Vocab]
            # 我们只需要最后一个位置: [Heads, 1, 1, Vocab]
            medusa_logits = correction_medusa_logits[:, :, -1:, :]  # [Heads, 1, 1, Vocab]
            
            # 6. 更新 token 计数
            new_token += al_val + 1

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
        
        # 生成结束后清理状态，防止内存泄漏
        reset_medusa_mode(self)
        # 清理 past_key_values 引用
        del past_key_values
        del past_key_values_data
        del current_length_data
        jt.gc()