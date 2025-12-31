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

# [修改类] GatedMedusaBlock
class GatedMedusaBlock(nn.Module):
    """
    包含预测头和状态更新机制的门控单元 (RNN-style)
    Logic:
        Logits = Head(S_k)
        if not last_head:
            Gate   = Sigmoid(Linear(S_k))
            S_new  = MLP(S_k)
            S_{k+1}= (1-Gate)*S_k + Gate*S_new
    """
    def __init__(self, hidden_size, vocab_size, is_last_head=False):
        super().__init__()
        self.is_last_head = is_last_head
        
        # 1. 预测层 (所有 Head 都有)
        self.head_linear = nn.Linear(hidden_size, vocab_size, bias=False)
        
        if not self.is_last_head:
            # 2. 门控层 (最后一个 Head 不需要)
            self.gate_linear = nn.Linear(hidden_size, hidden_size)
            
            # 3. 更新层 (最后一个 Head 不需要)
            self.update_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        
        # 初始化策略
        self.reset_parameters()

    def reset_parameters(self):
        # 预测头初始化为 0
        jt.init.constant_(self.head_linear.weight, 0.0)
        
        if not self.is_last_head:
            # 门控初始化：偏置设为 -2.0，使初始 Gate 接近 0
            jt.init.constant_(self.gate_linear.weight, 0.0)
            jt.init.constant_(self.gate_linear.bias, -2.0) 
            
            # Update MLP 初始化
            jt.init.constant_(self.update_mlp[0].weight, 0.0)
            jt.init.constant_(self.update_mlp[2].weight, 0.0)
            if self.update_mlp[0].bias is not None: jt.init.constant_(self.update_mlp[0].bias, 0.0)
            if self.update_mlp[2].bias is not None: jt.init.constant_(self.update_mlp[2].bias, 0.0)

    def execute(self, x):
        """
        Args:
            x (jt.Var): 当前状态 S_k [batch, seq_len, hidden_size]
        Returns:
            logits: 当前头的预测 [batch, seq_len, vocab_size]
            x_next: 下一时刻的状态 S_{k+1} (如果是最后一个 Head，返回 None)
        """
        # 1. 计算 Logits
        logits = self.head_linear(x)
        
        if self.is_last_head:
            # 如果是最后一个 Head，不需要计算下一个状态，直接返回 None
            return logits, None
            
        # 2. 计算门控 Z (范围 0~1)
        z = jt.sigmoid(self.gate_linear(x))
        
        # 3. 计算更新量
        h_new = self.update_mlp(x)
        
        # 4. 融合状态 (Gate 机制)
        # S_{k+1} = (1 - z) * S_k + z * h_new
        x_next = (1 - z) * x + z * h_new
        
        return logits, x_next

class MedusaModel(nn.Module):
    """
    Medusa 模型主体。
    它包装了一个基础的大语言模型 (base_model)，并附加了多个 Gated Medusa Blocks。
    """
    def __init__(self, config, base_model=None):
        super().__init__()
        self.config = config
        self.medusa_num_heads = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers
        self.base_model_name_or_path = config.base_model_name_or_path
        self.enable_lora_training = getattr(config, 'enable_lora_training', False)
        
        # 基础模型参数
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.base_model = base_model

        # [修改] 初始化 Block 时，标记最后一个 Block 为 is_last_head
        self.medusa_blocks = nn.ModuleList()
        for i in range(self.medusa_num_heads):
            is_last = (i == self.medusa_num_heads - 1)
            self.medusa_blocks.append(
                GatedMedusaBlock(self.hidden_size, self.vocab_size, is_last_head=is_last)
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
        **kwargs,
    ):
        """
        前向传播逻辑。
        """
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

        # Medusa 模式
        # 处理 Base Model 的梯度逻辑
        if self.enable_lora_training:
            # LoRA 训练：保留 Base Model 梯度
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
            
            # LoRA 模式下，Head 0 的输入需要保留梯度连接到 Base Model
            current_state = last_hidden_state.clone().float32()
            current_state.sync()

            medusa_logits = []
            # [修改] 串行执行 Medusa Blocks
            for block in self.medusa_blocks:
                logits, next_state = block(current_state)
                medusa_logits.append(logits)
                current_state = next_state # 传递状态到下一个 Head (最后一个为 None)
            
            medusa_logits_stack = jt.stack(medusa_logits, dim=0)
            
            if output_orig:
                orig_logits = base_outputs.logits
                saved_base_outputs = base_outputs
                return medusa_logits_stack, saved_base_outputs, orig_logits
            
            del base_outputs
            del last_hidden_state
            return medusa_logits_stack
            
        else:
            # 仅训练 Medusa Head (Base Model 冻结)
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
            
            # 仅切断 Base Model 到 Head 0 的梯度
            current_state = last_hidden_state.stop_grad()
            if current_state.dtype != "float32":
                current_state = current_state.float32()
            
            medusa_logits = []
            # [修改] 串行执行 Medusa Blocks
            for block in self.medusa_blocks:
                logits, next_state = block(current_state)
                medusa_logits.append(logits)
                current_state = next_state # 传递状态 (最后一个为 None)
            
            medusa_logits_stack = jt.stack(medusa_logits, dim=0)
            
            if output_orig:
                orig_logits = base_outputs.logits
                saved_base_outputs = base_outputs
                return medusa_logits_stack, saved_base_outputs, orig_logits
            
            return medusa_logits_stack

    def load_medusa_weights(self, weight_path):
        """
        专门加载 Medusa Head 的权重。增加对旧版权重和新版架构的兼容性检查。
        """
        print(f"Loading Medusa heads from {weight_path}")
        state_dict = jt.load(weight_path)
        
        new_state_dict = {}
        is_legacy = False
        
        for k, v in state_dict.items():
            if "medusa_head" in k:
                is_legacy = True
                break
        
        if is_legacy:
            print("WARNING: Detecting legacy Medusa weights (Independent Heads).")
            for k, v in state_dict.items():
                if "medusa_head" in k:
                    parts = k.split('.')
                    idx = int(parts[1])
                    if "1.weight" in k: # old output linear weight
                         new_key = f"medusa_blocks.{idx}.head_linear.weight"
                         new_state_dict[new_key] = v
            
            # [修改] 移除 strict=False
            self.load_parameters(new_state_dict)
        else:
            # [修改] 移除 strict=False
            self.medusa_blocks.load_parameters(state_dict)

    # def load_medusa_weights(self, weight_path):
    #     """
    #     专门加载 Medusa Head 的权重。增加对旧版权重和新版架构的兼容性检查。
    #     """
    #     print(f"Loading Medusa heads from {weight_path}")
    #     state_dict = jt.load(weight_path)
        
    #     new_state_dict = {}
    #     is_legacy = False
        
    #     for k, v in state_dict.items():
    #         if "medusa_head" in k:
    #             is_legacy = True
    #             break
        
    #     if is_legacy:
    #         print("WARNING: Detecting legacy Medusa weights (Independent Heads).")
    #         for k, v in state_dict.items():
    #             if "medusa_head" in k:
    #                 parts = k.split('.')
    #                 idx = int(parts[1])
    #                 if "1.weight" in k: # old output linear weight
    #                      new_key = f"medusa_blocks.{idx}.head_linear.weight"
    #                      new_state_dict[new_key] = v
    #         self.load_parameters(new_state_dict, strict=False)
    #     else:
    #         # 正常加载，使用 strict=False 以容忍旧权重中可能包含的最后一个 Head 的多余参数
    #         # 如果新权重已经去除了最后一个 Head 的 Gate 参数，那么完全匹配
    #         self.medusa_blocks.load_parameters(state_dict, strict=False)

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

        prefill_time = 0.0
        total_medusa_heads_time = 0.0
        total_tree_decoding_time = 0.0
        total_generate_candidates_time = 0.0
        total_evaluate_posterior_time = 0.0
        total_update_inputs_time = 0.0
        total_step_overhead_time = 0.0
        
        prefill_start = time.time()
        # 初始化阶段 (prefill)
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )
        jt.sync_all()
        prefill_time = time.time() - prefill_start
        
        self._prev_step_measured_time = 0.0

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
            
            # === [核心修改] 串行执行 Gated Medusa Blocks ===
            mh_start = time.time()
            
            # 初始状态 S_0 [1, 1, Hidden]
            current_state = last_hidden_state.stop_grad().float32()
            
            medusa_logits_list = []
            # 串行传递状态
            for block in self.medusa_blocks:
                logits_out, next_state = block(current_state)
                medusa_logits_list.append(logits_out)
                current_state = next_state 
            
            medusa_logits = jt.stack(medusa_logits_list, dim=0)  # [Heads, Batch, 1, Vocab]
            jt.sync_all()
            total_medusa_heads_time += time.time() - mh_start
            
            # Output processing (same as original)
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