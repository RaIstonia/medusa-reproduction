import jittor as jt
from jittor import nn
# from transformers import PretrainedConfig
import warnings
import copy
import time

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
    [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 1, 1], 
    [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], 
    [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], 
    [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [1, 6], [0, 7, 0]
    # 注意：已删除长度为4的路径 [0, 0, 0, 0] 和 [0, 0, 0, 1]
    # 因为训练时只使用了3个medusa heads，理论上最大深度应为3
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
        tokenizer = None,
        debug_callback = None
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        # 设置调试模式
        self._debug_mode = (debug_callback is not None)
        self._debug_callback = debug_callback
        
        if medusa_choices is None:
            medusa_choices = DEFAULT_MEDUSA_CHOICES
        
        # 根据 medusa_num_heads 过滤 medusa_choices，只保留深度 <= medusa_num_heads 的路径
        # 这确保树结构与实际训练的 medusa heads 数量匹配，避免 accept_length 超过理论最大值
        filtered_choices = [choice for choice in medusa_choices if len(choice) <= self.medusa_num_heads]
        if len(filtered_choices) != len(medusa_choices):
            medusa_choices = filtered_choices

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

        # 初始化性能统计变量
        prefill_time = 0.0  # prefill阶段的总时间（包含base model forward和medusa heads）
        total_medusa_heads_time = 0.0  # medusa heads预测token的总时间（不包括prefill）
        total_tree_decoding_time = 0.0  # base model tree decoding的总时间
        total_generate_candidates_time = 0.0  # generate_candidates的总时间
        total_evaluate_posterior_time = 0.0  # evaluate_posterior的总时间
        total_update_inputs_time = 0.0  # update_inference_inputs的总时间
        total_step_overhead_time = 0.0  # 每个step中未统计的时间（numpy转换、字符串处理、边界检查等）
        
        # 初始化（prefill）阶段：包含base model forward和medusa heads计算
        prefill_start = time.time()
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )
        jt.sync_all()
        prefill_time = time.time() - prefill_start
        
        # 初始化prev_step_measured_time（用于计算每个step中已统计的时间）
        self._prev_step_measured_time = 0.0

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
            step_start_time = time.time()
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
            gc_start = time.time()
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
            jt.sync_all()
            total_generate_candidates_time += time.time() - gc_start

            # (B) Tree Decoding
            # 保存 Tree Decoding 之前的 KV Cache 长度（用于后续 gather_and_reset）
            # 确保 prev_len_before_tree 是 Python int
            prev_len_before_tree = int(input_ids.shape[1])
            
            # 确保 medusa_mask 被设置（tree_decoding 需要它）
            # 注意：在第一次 prefill 后，medusa_mask 已经被 initialize_medusa 设置了
            # 但在回滚后的 Forward 中，我们清除了它，所以这里需要重新设置
            if self.base_model.model.medusa_mask is None:
                self.base_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"]
            
            # 统计tree decoding（base model forward）的时间
            td_start = time.time()
            tree_medusa_logits, tree_logits, original_tree_logits, original_tree_medusa_logits, tree_outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )
            jt.sync_all()
            total_tree_decoding_time += time.time() - td_start

            # (C) Evaluate Posterior
            # 检查是否需要返回调试信息（通过检查是否有debug_callback）
            return_debug = hasattr(self, '_debug_mode') and self._debug_mode
            # 获取当前序列（用于调试信息）
            current_seq_for_debug = input_ids[0] if return_debug else None
            ep_start = time.time()
            if return_debug:
                best_candidate, accept_length, debug_info = evaluate_posterior(
                    tree_logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast, tokenizer=tokenizer, return_debug_info=True, medusa_choices=medusa_choices, current_sequence=current_seq_for_debug
                )
                # 调用调试回调函数
                if hasattr(self, '_debug_callback') and self._debug_callback:
                    self._debug_callback(debug_info, idx, tokenizer)
            else:
                best_candidate, accept_length = evaluate_posterior(
                    tree_logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast, tokenizer=tokenizer, medusa_choices=medusa_choices, current_sequence=None
                )
            jt.sync_all()
            total_evaluate_posterior_time += time.time() - ep_start

            # (D) Efficient Update (关键优化)
            # 核心优化：从 Tree Decoding 的结果中直接提取最后一个被接受 token 的 hidden state
            # 不再需要 Correction Step Forward，因为 hidden state 已经在 tree decoding 中计算好了
            
            # 调用新的 update_inference_inputs，返回 last_hidden_state 和 medusa_logits
            ui_start = time.time()
            input_ids, last_hidden_state, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids=input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=medusa_buffers["retrieve_indices"],
                tree_logits=original_tree_logits,        # 传入原始 tree_logits [Batch, Tree_Size, Vocab]
                tree_medusa_logits=original_tree_medusa_logits, # 传入原始 tree_medusa_logits [Heads, Batch, Tree_Size, Vocab]
                tree_outputs=tree_outputs,               # 传入 tree_outputs 以提取 hidden_states
                new_token=new_token,
                past_key_values=past_key_values,         # 传入对象列表以进行 Gather
                prev_len_before_tree=prev_len_before_tree  # 传入基准长度
            )
            jt.sync_all()
            total_update_inputs_time += time.time() - ui_start
            
            # === [优化] 直接使用 update_inference_inputs 返回的 medusa_logits ===
            # 这里的 medusa_logits 已经是 [Heads, Vocab] 格式，从 tree_medusa_logits 中提取
            # 需要调整为 [Heads, Batch, 1, Vocab] 格式用于 generate_candidates
            if medusa_logits.ndim == 2:
                # [Heads, Vocab] -> [Heads, Batch, 1, Vocab]
                medusa_logits = medusa_logits.unsqueeze(1).unsqueeze(2)
            elif medusa_logits.ndim == 3:
                # [Heads, Batch, Vocab] -> [Heads, Batch, 1, Vocab]
                medusa_logits = medusa_logits.unsqueeze(2)
            
            # 更新统计时间（medusa heads 计算已包含在 update_inputs_time 中）
            total_medusa_heads_time = total_update_inputs_time  # 合并时间统计
            
            # logits 已经由 update_inference_inputs 从 tree_logits 中提取
            # logits shape: [Batch, 1, Vocab]
            
            # === [性能优化] 极简的 Loop 结尾 ===
            # 1. 计算 ids（用于统计 token 数量），但仅在需要时进行解码
            # 在 fast 模式下，我们只计算 ids 但不进行字符串解码
            output_start = time.time()
            curr_ids = input_ids[0, input_len:].numpy().tolist()
            output_time = time.time() - output_start
            
            # 2. 仅在非 fast 模式或 debug 模式下才进行字符串解码
            if not fast or self._debug_mode:
                if tokenizer is not None:
                    text = tokenizer.decode(curr_ids, skip_special_tokens=True)
                else:
                    text = None
            else:
                # 测速模式：跳过字符串解码（但保留 ids 用于统计）
                text = None
            
            # 2. 高效的 EOS 检查
            # 不转换整个 list，直接检查最后一个 token
            eos_detected = False
            if tokenizer is not None and tokenizer.eos_token_id is not None:
                # 只获取最后一个 token 的值 (Scalar Sync，比全量 Sync 快得多)
                try:
                    last_token_id = int(input_ids[0, -1].item())
                    if last_token_id == tokenizer.eos_token_id:
                        eos_detected = True
                except:
                    # 如果上面的方法失败，跳过 EOS 检查（在 fast 模式下）
                    pass
            
            # 计算当前step的总时间和已统计时间
            step_end_time = time.time()
            step_total_time = step_end_time - step_start_time
            
            # 计算本次step中已统计的时间
            # 记录本次step开始时的累计时间值
            if idx == 0:
                # 第一个step，记录初始累计时间
                prev_measured_time = 0.0
            else:
                # 使用上一次的累计时间
                prev_measured_time = getattr(self, '_prev_step_measured_time', 0.0)
            
            # 当前累计的已统计时间（不包括output_time，因为它是本次step新增的）
            current_measured_time = (
                total_generate_candidates_time +
                total_tree_decoding_time +
                total_evaluate_posterior_time +
                total_update_inputs_time +
                total_medusa_heads_time
            )
            
            # 本次step中已统计的时间 = 当前累计时间 - 上次累计时间 + 本次output时间
            step_measured_time = current_measured_time - prev_measured_time + output_time
            
            # 本次step中未统计的时间（边界检查、EOS检查等）
            step_overhead = step_total_time - step_measured_time
            total_step_overhead_time += step_overhead
            
            # 保存当前累计时间，供下次使用
            self._prev_step_measured_time = current_measured_time
            
            # yield 结果
            # 基础信息（所有模式都需要）
            yield_dict = {
                "ids": curr_ids,          # 测试脚本需要统计 len(ids)，所有模式都需要
                "accept_length": accept_length, # 用于统计加速效率
                "step": idx,              # 当前步数 (idxs)
                # 性能统计信息（累计值）
                "prefill_time": prefill_time,
                "total_medusa_heads_time": total_medusa_heads_time,
                "total_tree_decoding_time": total_tree_decoding_time,
                "total_generate_candidates_time": total_generate_candidates_time,
                "total_evaluate_posterior_time": total_evaluate_posterior_time,
                "total_update_inputs_time": total_update_inputs_time,
                "total_step_overhead_time": total_step_overhead_time,
            }
            
            # 仅在非 fast 模式或 debug 模式下添加 text
            if not fast or self._debug_mode:
                yield_dict["text"] = text
            
            yield yield_dict
            
            # Check EOS
            if eos_detected or (tokenizer is not None and tokenizer.eos_token_id is not None and tokenizer.eos_token_id in curr_ids):
                break
        
        loop_end_time = time.time()
        # 计算循环总时间（包括所有未统计的开销）
        # 注意：这个时间会在外部（test_medusa_benchmark.py）与generation_time对比来计算剩余时间
        
        # 生成结束后清理状态，防止内存泄漏
        reset_medusa_mode(self)
        # 清理 past_key_values 引用
        del past_key_values
        del past_key_values_data
        del current_length_data
        jt.gc()