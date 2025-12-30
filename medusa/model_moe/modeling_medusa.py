import jittor as jt
from jittor import nn
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
    Medusa 配置类，新增 MoE 相关超参数。
    """
    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        enable_lora_training=False,
        moe_num_experts=4,  # [新增] 专家数量
        moe_top_k=1,        # [新增] 激活专家数量
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.enable_lora_training = enable_lora_training
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k

# [新增] MedusaMoEBlock: MoE 风格的门控单元
class MedusaMoEBlock(nn.Module):
    """
    MoE 风格的门控单元。
    Structure:
        1. Logits = Head(S_k)
        2. Router -> TopK Experts
        3. ExpertOutput = WeightedSum(Experts(S_k))
        4. S_{k+1} = (1-Gate)*S_k + Gate*ExpertOutput
    """
    def __init__(self, config, is_last_head=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.is_last_head = is_last_head

        # 1. 预测头 (所有 Head 都有)
        self.head_linear = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        if not self.is_last_head:
            # 2. 门控层 (Gate)
            self.gate_linear = nn.Linear(self.hidden_size, self.hidden_size)
            
            # 3. Router (用于选择专家)
            self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
            
            # 4. Experts (专家组, 替代原来的 Update MLP)
            # 每个 Expert 也是一个简单的 MLP: Linear -> SiLU -> Linear
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size)
                )
                for _ in range(self.num_experts)
            ])
        
        self.reset_parameters()

    def reset_parameters(self):
        jt.init.constant_(self.head_linear.weight, 0.0)
        
        if not self.is_last_head:
            # Gate 初始化：偏向于保留旧状态
            jt.init.constant_(self.gate_linear.weight, 0.0)
            jt.init.constant_(self.gate_linear.bias, -2.0)
            
            # Router 初始化：需要足够的随机性以防止初始坍缩
            jt.init.xavier_uniform_(self.router.weight)
            
            # Experts 初始化：接近 Identity 或 Zero
            for expert in self.experts:
                jt.init.constant_(expert[0].weight, 0.0)
                jt.init.constant_(expert[2].weight, 0.0)
                if expert[0].bias is not None: jt.init.constant_(expert[0].bias, 0.0)
                if expert[2].bias is not None: jt.init.constant_(expert[2].bias, 0.0)

    def _compute_moe_output(self, x, selected_experts, weights):
        """
        计算 MoE 输出的辅助函数
        Args:
            x: [batch, seq_len, hidden]
            selected_experts: [batch, seq_len, top_k] (Indices)
            weights: [batch, seq_len, top_k] (Normalized Weights)
        """
        expert_output = jt.zeros_like(x)
        
        # 优化：Top-1 情况下的循环计算
        # 相比于 gather 整个 parameter table，分别计算激活的 expert 并 mask 融合在显存上更友好
        if self.top_k == 1:
            expert_idx = selected_experts[..., 0] # [batch, seq_len]
            weight = weights[..., 0].unsqueeze(-1) # [batch, seq_len, 1]
            
            for i in range(self.num_experts):
                # 创建 mask: 哪些 token 选中了 expert i
                mask = (expert_idx == i)
                if mask.any():
                    # 只有当该 expert 被选中时才计算图
                    out = self.experts[i](x)
                    # 使用 where 进行融合
                    expert_output = jt.where(mask.unsqueeze(-1), out, expert_output)
            
            return expert_output * weight
        else:
            # 通用 Top-k 实现
            for k_idx in range(self.top_k):
                idx_k = selected_experts[..., k_idx] # [batch, seq_len]
                w_k = weights[..., k_idx].unsqueeze(-1) # [batch, seq_len, 1]
                
                term_k = jt.zeros_like(x)
                for i in range(self.num_experts):
                    mask = (idx_k == i)
                    if mask.any():
                        out = self.experts[i](x)
                        term_k = jt.where(mask.unsqueeze(-1), out, term_k)
                
                expert_output += term_k * w_k
                
            return expert_output

    def execute(self, x):
        """
        Returns:
            logits: 预测结果
            x_next: 下一步状态
            router_logits: 路由器的原始输出 (用于计算 aux loss)
        """
        logits = self.head_linear(x)
        
        if self.is_last_head:
            return logits, None, None
            
        # 1. Router Forward
        router_logits = self.router(x)
        
        # 2. Top-K Gating
        routing_weights = jt.nn.softmax(router_logits, dim=-1)
        weights, selected_experts = jt.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights (sum to 1)
        weights = weights / (weights.sum(dim=-1, keepdims=True) + 1e-9)
        
        # 3. MoE Computation
        expert_output = self._compute_moe_output(x, selected_experts, weights)
        
        # 4. Gated Update
        z = jt.sigmoid(self.gate_linear(x))
        x_next = (1 - z) * x + z * expert_output
        
        return logits, x_next, router_logits

class MedusaModel(nn.Module):
    def __init__(self, config, base_model=None):
        super().__init__()
        self.config = config
        self.medusa_num_heads = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers
        self.base_model_name_or_path = config.base_model_name_or_path
        self.enable_lora_training = getattr(config, 'enable_lora_training', False)
        
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.base_model = base_model

        # [修改] 使用 MedusaMoEBlock
        self.medusa_blocks = nn.ModuleList()
        for i in range(self.medusa_num_heads):
            is_last = (i == self.medusa_num_heads - 1)
            self.medusa_blocks.append(
                MedusaMoEBlock(config, is_last_head=is_last)
            )
        
        self.medusa_buffers = None
        self.medusa_choices = None

    def set_base_model(self, base_model):
        self.base_model = base_model

    def get_tokenizer(self):
        if hasattr(self, 'tokenizer'):
            return self.tokenizer
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
        return_medusa_logits=True,
        return_router_logits=False, # [新增] 控制是否返回 router logits
        **kwargs,
    ):
        if not medusa_forward:
            if self.base_model is None:
                raise RuntimeError("Base model is not initialized.")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                return_dict=True,
                **kwargs
            )

        if self.enable_lora_training:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            last_hidden_state = base_outputs.hidden_states[-1]
            
            if not return_medusa_logits:
                if output_orig:
                    return None, base_outputs, base_outputs.logits
                return None
            
            current_state = last_hidden_state.clone().float32()
            current_state.sync()

            medusa_logits = []
            router_logits_list = [] # 收集 Router Logits
            
            for block in self.medusa_blocks:
                logits, next_state, router_logits = block(current_state)
                medusa_logits.append(logits)
                if router_logits is not None:
                    router_logits_list.append(router_logits)
                current_state = next_state
            
            medusa_logits_stack = jt.stack(medusa_logits, dim=0)
            
            if output_orig:
                orig_logits = base_outputs.logits
                saved_base_outputs = base_outputs
                if return_router_logits:
                    return medusa_logits_stack, saved_base_outputs, orig_logits, router_logits_list
                return medusa_logits_stack, saved_base_outputs, orig_logits
            
            del base_outputs
            del last_hidden_state
            
            if return_router_logits:
                return medusa_logits_stack, router_logits_list
            return medusa_logits_stack
            
        else:
            with jt.no_grad():
                base_outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs,
                )
            
            last_hidden_state = base_outputs.hidden_states[-1]

            if not return_medusa_logits:
                if output_orig:
                    return None, base_outputs, base_outputs.logits
                return None
            
            current_state = last_hidden_state.stop_grad()
            if current_state.dtype != "float32":
                current_state = current_state.float32()
            
            medusa_logits = []
            router_logits_list = []
            
            for block in self.medusa_blocks:
                logits, next_state, router_logits = block(current_state)
                medusa_logits.append(logits)
                if router_logits is not None:
                    router_logits_list.append(router_logits)
                current_state = next_state
            
            medusa_logits_stack = jt.stack(medusa_logits, dim=0)
            
            if output_orig:
                orig_logits = base_outputs.logits
                saved_base_outputs = base_outputs
                if return_router_logits:
                     return medusa_logits_stack, saved_base_outputs, orig_logits, router_logits_list
                return medusa_logits_stack, saved_base_outputs, orig_logits
            
            if return_router_logits:
                return medusa_logits_stack, router_logits_list
            return medusa_logits_stack

    def load_medusa_weights(self, weight_path):
        print(f"Loading Medusa heads from {weight_path}")
        state_dict = jt.load(weight_path)
        # 加载时允许不匹配（旧权重可能没有 experts）
        # 如果是第一次从 Gated RNN 迁移到 MoE，建议从头训练或仅加载 head_linear
        self.medusa_blocks.load_parameters(state_dict, strict=False)

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
        # ... (基本逻辑与之前相同) ...
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        self._debug_mode = (debug_callback is not None)
        self._debug_callback = debug_callback
        
        if medusa_choices is None:
            medusa_choices = DEFAULT_MEDUSA_CHOICES
        
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
        
        # --- 省略部分初始化代码，保持原样 ---
        prefill_time = 0.0
        total_medusa_heads_time = 0.0
        total_tree_decoding_time = 0.0
        total_generate_candidates_time = 0.0
        total_evaluate_posterior_time = 0.0
        total_update_inputs_time = 0.0
        total_step_overhead_time = 0.0
        self._prev_step_measured_time = 0.0
        
        prefill_start = time.time()
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )
        jt.sync_all()
        prefill_time = time.time() - prefill_start

        new_token = 0
        input_len = input_ids.shape[1]
        
        if self.base_model and hasattr(self.base_model, 'config'):
            max_model_len = getattr(self.base_model.config, "max_position_embeddings", 2048)
        else:
            max_model_len = 2048
        
        if medusa_choices is not None:
            max_tree_depth = max(len(choice) for choice in medusa_choices) if medusa_choices else 0
            safe_margin = max(64, max_tree_depth * 2)
        else:
            safe_margin = 64

        for idx in range(max_steps):
            step_start_time = time.time()
            current_seq_len = input_ids.shape[1]
            if current_seq_len + safe_margin >= max_model_len:
                if tokenizer is not None:
                    curr_ids = input_ids[0, input_len:].numpy().tolist()
                    text = tokenizer.decode(curr_ids, skip_special_tokens=True)
                    yield {"text": text, "ids": curr_ids, "accept_length": 0, "step": idx}
                else:
                    curr_ids = input_ids[0, input_len:].numpy().tolist()
                    yield {"ids": curr_ids, "accept_length": 0, "step": idx}
                break

            # (A) Generate Candidates
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
            prev_len_before_tree = int(input_ids.shape[1])
            if self.base_model.model.medusa_mask is None:
                self.base_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"]
            
            td_start = time.time()
            _, tree_logits, original_tree_logits, _, tree_outputs = tree_decoding(
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
            return_debug = hasattr(self, '_debug_mode') and self._debug_mode
            current_seq_for_debug = input_ids[0] if return_debug else None
            ep_start = time.time()
            if return_debug:
                best_candidate, accept_length, debug_info = evaluate_posterior(
                    tree_logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast, tokenizer=tokenizer, return_debug_info=True, medusa_choices=medusa_choices, current_sequence=current_seq_for_debug,
                    retrieve_indices=medusa_buffers["retrieve_indices"]
                )
                if hasattr(self, '_debug_callback') and self._debug_callback:
                    self._debug_callback(debug_info, idx, tokenizer)
            else:
                best_candidate, accept_length = evaluate_posterior(
                    tree_logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast, tokenizer=tokenizer, medusa_choices=medusa_choices, current_sequence=None,
                    retrieve_indices=medusa_buffers["retrieve_indices"]
                )
            jt.sync_all()
            total_evaluate_posterior_time += time.time() - ep_start

            # (D) Efficient Update
            ui_start = time.time()
            input_ids, last_hidden_state, logits, _, new_token = update_inference_inputs(
                input_ids=input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=medusa_buffers["retrieve_indices"],
                tree_logits=original_tree_logits,
                tree_medusa_logits=None,
                tree_outputs=tree_outputs,
                new_token=new_token,
                past_key_values=past_key_values,
                prev_len_before_tree=prev_len_before_tree
            )
            jt.sync_all()
            total_update_inputs_time += time.time() - ui_start
            
            # === [修改] 串行执行 MoE Block，忽略 router logits ===
            mh_start = time.time()
            
            current_state = last_hidden_state.stop_grad().float32()
            
            medusa_logits_list = []
            for block in self.medusa_blocks:
                logits_out, next_state, _ = block(current_state) # Ignore router_logits
                medusa_logits_list.append(logits_out)
                current_state = next_state 
            
            medusa_logits = jt.stack(medusa_logits_list, dim=0)
            jt.sync_all()
            total_medusa_heads_time += time.time() - mh_start
            
            # Output processing...
            # ... (保持原样) ...
            output_start = time.time()
            curr_ids = input_ids[0, input_len:].numpy().tolist()
            output_time = time.time() - output_start
            
            if not fast or self._debug_mode:
                if tokenizer is not None:
                    text = tokenizer.decode(curr_ids, skip_special_tokens=True)
                else:
                    text = None
            else:
                text = None
            
            eos_detected = False
            if tokenizer is not None and tokenizer.eos_token_id is not None:
                try:
                    last_token_id = int(input_ids[0, -1].item())
                    if last_token_id == tokenizer.eos_token_id:
                        eos_detected = True
                except:
                    pass
            
            # ... (时间统计代码保持原样) ...
            step_end_time = time.time()
            step_total_time = step_end_time - step_start_time
            
            if idx == 0:
                prev_measured_time = 0.0
            else:
                prev_measured_time = getattr(self, '_prev_step_measured_time', 0.0)
            
            current_measured_time = (
                total_generate_candidates_time +
                total_tree_decoding_time +
                total_evaluate_posterior_time +
                total_update_inputs_time +
                total_medusa_heads_time
            )
            
            step_measured_time = current_measured_time - prev_measured_time + output_time
            step_overhead = step_total_time - step_measured_time
            total_step_overhead_time += step_overhead
            self._prev_step_measured_time = current_measured_time
            
            yield_dict = {
                "ids": curr_ids,
                "accept_length": accept_length,
                "step": idx,
                "prefill_time": prefill_time,
                "total_medusa_heads_time": total_medusa_heads_time,
                "total_tree_decoding_time": total_tree_decoding_time,
                "total_generate_candidates_time": total_generate_candidates_time,
                "total_evaluate_posterior_time": total_evaluate_posterior_time,
                "total_update_inputs_time": total_update_inputs_time,
                "total_step_overhead_time": total_step_overhead_time,
            }
            if not fast or self._debug_mode:
                yield_dict["text"] = text
            yield yield_dict
            
            if eos_detected or (tokenizer is not None and tokenizer.eos_token_id is not None and tokenizer.eos_token_id in curr_ids):
                break
        
        reset_medusa_mode(self)
        del past_key_values
        del past_key_values_data
        del current_length_data
        jt.gc()