# medusa/model/modeling_llama_kv.py

import math
import jittor as jt
from jittor import nn
# from transformers import PretrainedConfig
# from transformers.activations import ACT2FN
# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
# from transformers.utils import logging
# from transformers.models.llama.configuration_llama import LlamaConfig
import logging
from typing import Optional, Tuple, Union, List
from medusa.model.llama_src.llama_config import LlamaConfig
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class PretrainedConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        import copy
        return copy.deepcopy(self.__dict__)

@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: jt.Var = None
    past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None

@dataclass
class CausalLMOutputWithPast:
    loss: Optional[jt.Var] = None
    logits: jt.Var = None
    past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None

@dataclass
class SequenceClassifierOutputWithPast:
    loss: Optional[jt.Var] = None
    logits: jt.Var = None
    past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None


def _make_causal_mask(
    input_ids_shape: list, dtype: str, past_key_values_length: int = 0
):

    bsz, tgt_len = input_ids_shape
    # min_val for float16/32
    min_val = -65504.0 if dtype == "float16" else -1e30
    
    mask = jt.full((tgt_len, tgt_len), min_val, dtype=dtype)
    mask_cond = jt.arange(mask.shape[-1])

    is_causal = mask_cond.unsqueeze(0) < (mask_cond + 1).unsqueeze(1)
    mask = jt.where(is_causal, jt.zeros_like(mask), mask)

    if past_key_values_length > 0:
        mask = jt.concat([jt.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], dim=-1)
    
    # [bsz, 1, tgt_len, tgt_len + past_kv_len]
    return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: jt.Var, dtype: str, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Here, what we send in is a stardard mask array containing 0 and 1, and we need to convert it into the final addition mask.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgt_len, src_len).astype(dtype)
    inverted_mask = 1.0 - expanded_mask
    
    min_val = -65504.0 if dtype == "float16" else -1e30
    
    return jt.where(inverted_mask.astype("bool"), min_val, inverted_mask)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = jt.ones(hidden_size)
        self.variance_epsilon = eps

    def execute(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float32()
        variance = hidden_states.pow(2).mean(-1, keepdims=True)
        hidden_states = hidden_states * jt.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).astype(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Jittor 不需要 register_buffer，直接作为成员变量即可，会自动被 save/load 处理
        inv_freq = 1.0 / (self.base ** (jt.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq
        
        # Build cos/sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=jt.float32
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = jt.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = jt.einsum("i,j->ij", t, self.inv_freq)
        emb = jt.concat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(0).astype(dtype)
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(0).astype(dtype)

    def execute(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].astype(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].astype(x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jt.concat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos/sin: [1, 1, seq_len, dim]
    # position_ids: [bs, seq_len]
    
    # Jittor indexing: squeeze first to match implementation
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    
    # Advanced indexing
    # cos[position_ids] -> [bs, seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = ACT2FN[config.hidden_act]
        if config.hidden_act == "silu":
            self.act = nn.SiLU()
        elif config.hidden_act == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.SiLU() # TODO: 实现更多的支持 包括错误处理

    def execute(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: jt.Var, n_rep: int) -> jt.Var:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # Jittor expand logic
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self._init_rope()

    def _init_rope(self):
        # Simplified RoPE init for Jittor
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        past_key_value: Optional[Tuple[jt.Var]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, Optional[jt.Var], Optional[Tuple[jt.Var]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # Assuming past_key_value[0] is (bsz, num_heads, past_len, head_dim)
            kv_seq_len += past_key_value[0].shape[-2]
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [MODIFIED] KVCache Integration
        # 与原版 PyTorch 保持一致的处理方式
        if past_key_value is not None:
            if hasattr(past_key_value[0], 'cat'): 
                # KVCache 模式：使用预分配的共享内存
                key_states = past_key_value[0].cat(key_states, dim=2)
                value_states = past_key_value[1].cat(value_states, dim=2)
                # [关键修复] 原版 PyTorch 在使用 KVCache.cat() 后设 past_key_value = None
                # 这是因为 KV 已经写入共享的 past_key_values_data，不需要再返回
                past_key_value = None
            else:
                # 标准 concat 模式
                key_states = jt.concat([past_key_value[0], key_states], dim=2)
                value_states = jt.concat([past_key_value[1], value_states], dim=2)
                past_key_value = (key_states, value_states) if use_cache else None

        # 保存用于返回的 present_key_value（在 repeat_kv 之前）
        # 注意：如果 past_key_value 是 None 且 use_cache 是 True，需要返回当前的 key_states 和 value_states
        if use_cache and past_key_value is None:
            # 第一次调用时，返回当前的 key_states 和 value_states 作为 present_key_value
            # 注意：这里保存的是 repeat_kv 之前的原始 key_states 和 value_states
            present_key_value = (key_states, value_states)
        else:
            present_key_value = past_key_value

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = jt.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = nn.softmax(attn_weights, dim=-1)
        # Upcast is usually handled by Jittor internally or exact dtype matching
        
        attn_output = jt.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Fallback to standard attention for stability
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        past_key_value: Optional[Tuple[jt.Var]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, Optional[Tuple[jt.Var, jt.Var]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # init weights is implicit or handled by loader in Jittor, 
        # but to match logic we can have an empty post_init
        
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add medusa mask
        # 这是 Medusa 的核心逻辑：如果存在 medusa_mask，则将其叠加到 attention mask 上
        if hasattr(self, "medusa_mask") and self.medusa_mask is not None:
            # [CRITICAL FIX] 如果 combined_attention_mask 为 None，说明序列长度为 1，不需要应用 medusa_mask
            # medusa_mask 只在序列长度 > 1 时才有意义（tree decoding 阶段）
            if combined_attention_mask is not None:
                # [CRITICAL FIX] 必须先 clone，否则对 expand 出来的 view 进行 setitem 会导致 CUDA Illegal Address
                combined_attention_mask = combined_attention_mask.clone()
                
                medusa_mask = self.medusa_mask # [bs, medusa_len]
                medusa_len = medusa_mask.shape[-1]
                
                # 获取目标切片
                # 注意：如果 combined_attention_mask 比 medusa_mask 小，这里会报错。
                # 但在 tree_decoding 阶段，combined_attention_mask 长度至少为 medusa_len。
                # 添加检查，确保 combined_attention_mask 的长度足够
                seq_len = combined_attention_mask.shape[-1]
                if seq_len >= medusa_len:
                    target_slice = combined_attention_mask[:, :, -medusa_len:, -medusa_len:]
                    
                    min_val = combined_attention_mask.min()
                    
                    # 构造新的 mask 片段
                    masked_slice = jt.where(medusa_mask == 0, min_val, target_slice)
                    
                    # 写回 (SetItem)
                    combined_attention_mask[:, :, -medusa_len:, -medusa_len:] = masked_slice

        return combined_attention_mask

    def execute(
        self,
        input_ids: jt.Var = None,
        attention_mask: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        past_key_values=None, 
        inputs_embeds: Optional[jt.Var] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # 获取 past length
        if past_key_values is not None:
            # 假设 standard tuple 结构: ((k,v), (k,v)...)
            # past_key_values[0][0] -> k tensor
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = jt.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        # Medusa 运行时需要保存这些属性
        self.attention_mask = attention_mask
        self.position_ids = position_ids

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        # 返回类似 HuggingFace 的命名元组结构
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def execute(
        self,
        input_ids: jt.Var = None,
        attention_mask: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        past_key_values=None,
        inputs_embeds: Optional[jt.Var] = None,
        labels: Optional[jt.Var] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float32() # Ensure FP32 for stability

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = nn.cross_entropy_loss(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + (outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )