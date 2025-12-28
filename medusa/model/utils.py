import jittor as jt
from jittor import nn
import numpy as np

TOPK = 10 

def pad_path(path, length, pad_value=-2):
    """
    Python list padding logic (Identical to Torch version).
    """
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure.
    """
    # 纯 Python 逻辑：排序和统计
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # --- Attention Mask ---
    # Jittor 不支持直接 tensor[idx_list] = 1 这种高级索引赋值初始化，建议先用 numpy 或 python list 构建
    # 或者小心使用 scatter。考虑到这是初始化阶段，不影响性能，我们可以用 numpy 过渡，或者纯 Jittor 构建。
    # 为了纯粹性，这里用 Jittor 构建，但逻辑稍微拆解。
    
    # 实际上，mask 结构比较简单，可以直接操作 Var。
    # 初始化对角矩阵
    # medusa_attn_mask = jt.init.eye(medusa_len, dtype="float32") # Jittor eye return float usually
    # 但 Jittor 似乎没有 jt.eye，用 jt.diag?
    # 我们可以直接生成 np array 然后转 Jittor Var，这是最稳妥的 buffer 生成方式。
    import numpy as np
    
    mask_np = np.eye(medusa_len, dtype=np.float32)
    mask_np[:, 0] = 1
    
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            
            # mask_np[row, cols] = 1
            row_idx = j + start + 1
            mask_np[row_idx, ancestor_idx] = 1
        start += depth_counts[i]
    
    medusa_attn_mask = jt.array(mask_np)

    # --- Tree Indices ---
    tree_indices_np = np.zeros(medusa_len, dtype=np.int64)
    tree_indices_np[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # Medusa choice content are token offsets?
            tree_indices_np[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]
    
    medusa_tree_indices = jt.array(tree_indices_np)

    # --- Position IDs ---
    position_ids_np = np.zeros(medusa_len, dtype=np.int64)
    start = 0
    for i in range(len(depth_counts)):
        position_ids_np[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]
    
    medusa_position_ids = jt.array(position_ids_np)

    # --- Retrieval Indices ---
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices_list = [pad_path(path, max_length) for path in retrieve_indices_nest]
    
    # Convert to Tensor
    retrieve_indices = jt.array(retrieve_indices_list).int64() # [N, MaxLen]
    retrieve_indices = retrieve_indices + 1
    # Concat zero column
    zeros_col = jt.zeros((retrieve_indices.shape[0], 1), dtype="int64")
    retrieve_indices = jt.concat([zeros_col, retrieve_indices], dim=1)

    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
    }
    
    # Jittor Var 已经在正确的 device 上 (自动管理)，不需要手动 to(device)
    # 但如果为了兼容接口，可以保留。
    # 这里我们不需要 clone().to()，直接返回即可。
    return medusa_buffers


def initialize_medusa(input_ids, model, medusa_attn_mask, past_key_values):
    """
    Initializes the Medusa structure.
    """
    # Forward pass
    medusa_logits, outputs, logits = model(
        input_ids, 
        past_key_values=past_key_values, 
        output_orig=True, 
        medusa_forward=True
    )
    
    # Set mask to base model
    # 注意：Jittor 模型通常属性是动态的，但要注意 LlamaModel 是否有这个属性
    # 我们在 Phase 2 已经在 LlamaModel 里预埋了 hasattr(self, "medusa_mask") 的检查
    model.base_model.model.medusa_mask = medusa_attn_mask
    return medusa_logits, logits


def reset_medusa_mode(model):
    model.base_model.model.medusa_mask = None
    if hasattr(model.base_model.model, "medusa_mode"):
        model.base_model.model.medusa_mode = None

def reset_past_key_values(passed_key_values):
    """
    Reset KV Cache lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            # 使用正确的方式更新长度
            passed_key_values[i][j]._set_current_length(0)
    return passed_key_values

# --- Sampling Utils ---

def get_nucleus_one_token(logit, temperature, top_p):
    """
    Nucleus sampling.
    """
    if top_p >= 1:
        # Simple multinomial
        probs = nn.softmax(logit / temperature, dim=-1)
        return jt.multinomial(probs, 1)
        
    logit = logit / temperature
    probs = nn.softmax(logit, dim=-1)
    
    # Sort
    sorted_logits, sorted_indices = jt.argsort(probs, descending=True)
    cum_probs = jt.cumsum(sorted_logits, dim=-1)
    
    sorted_indices_to_remove = cum_probs > top_p
    # Shift right: [..., 1:] = [..., :-1]
    # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    # sorted_indices_to_remove[..., 0] = 0
    # Jittor Shift Implementation:
    shift_mask = jt.concat([
        jt.zeros((sorted_indices_to_remove.shape[0], 1), dtype="bool"),
        sorted_indices_to_remove[:, :-1]
    ], dim=1)
    
    # Map back to original indices
    # indices_to_remove = shift_mask.scatter(dim=1, index=sorted_indices, src=shift_mask)
    # Jittor scatter:
    # 构造一个全False的mask
    indices_to_remove = jt.zeros_like(shift_mask).bool()
    # scatter_ usually works, or we use advanced indexing if supported
    # Jittor's scatter: jt.scatter(dim, index, src, reduce=None) -> returns new tensor
    # src must be broadcastable.
    # Note: Jittor scatter behavior might slightly differ. Let's use numpy-style if possible or scatter.
    # But scatter is safe here.
    
    # 实际上，我们可以不需要 scatter 回去，直接在 sorted 空间采样？
    # 不行，后续逻辑依赖原始 logit 顺序置 -inf。
    
    # 替代方案：直接用 scatter
    # indices_to_remove[b, sorted_indices[b, i]] = shift_mask[b, i]
    # Jittor scatter: output = input.scatter(dim, index, src)
    # 这里的 input 是 zeros
    indices_to_remove = indices_to_remove.scatter(1, sorted_indices, shift_mask)

    # logit[indices_to_remove] = -inf
    # Jittor where
    logit = jt.where(indices_to_remove, -1e30, logit)
    
    sampled_tokens = jt.multinomial(nn.softmax(logit, dim=-1), 1)
    return sampled_tokens


def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    logit = logit / temperature
    probs = nn.softmax(logit, dim=-1)
    
    entropy = -jt.sum(probs * jt.log(probs + 1e-5), dim=-1)
    
    threshold = jt.minimum(
        jt.ones_like(entropy) * posterior_threshold,
        jt.exp(-entropy) * posterior_alpha,
    )
    
    indices_to_remove = probs < threshold.unsqueeze(-1)
    
    logit = jt.where(indices_to_remove, -1e30, logit)
    
    sampled_tokens = jt.multinomial(nn.softmax(logit, dim=-1), 1)
    return sampled_tokens


def generate_candidates(medusa_logits, logits, tree_indices, retrieve_indices, 
                        temperature=0, posterior_threshold=0.3, posterior_alpha=0.09, 
                        top_p=0.8, sampling='typical', fast=False,
                        suppress_special_tokens=True):
    """
    Generate candidates based on provided logits and indices.
    
    Args:
        suppress_special_tokens: If True, suppress BOS/EOS tokens from being selected.
                                This helps avoid model quirks where special tokens get high probability.
    """
    # === [重要修复] 抑制特殊 token ===
    # Base model 有时会给 BOS token (<s>) 很高的概率，导致生成跑偏
    # 参考：PyTorch transformers 的 LogitsProcessor
    if suppress_special_tokens:
        # Clone logits to avoid modifying original
        base_logits = logits[:, -1].clone()
        medusa_logits_last = medusa_logits[:, 0, -1].clone()
        
        # Suppress BOS token (usually token 1) and PAD token (usually token 0)
        # 注意：不完全抑制EOS token，允许它在适当时候被生成以自然结束生成
        # 只抑制BOS和PAD，因为它们在生成过程中不应该出现
        special_tokens_to_suppress = [0, 1]  # <unk>, <s>
        neg_inf = jt.float32(-1e9)  # Use large negative instead of -inf for numerical stability
        
        for tok_id in special_tokens_to_suppress:
            base_logits[:, tok_id] = neg_inf
            medusa_logits_last[:, tok_id] = neg_inf
        
        # 对于EOS token，不完全抑制，而是降低其概率（允许在适当时候生成）
        # 这样可以允许模型在完成回答后自然结束
        # EOS token通常是2，但为了安全，我们检查一下
        # 实际上，如果EOS概率足够高，typical acceptance会接受它
    else:
        base_logits = logits[:, -1]
        medusa_logits_last = medusa_logits[:, 0, -1]
    
    # 1. Base Model Candidate
    if temperature == 0 or fast:
        # [MODIFIED] Use topk instead of argmax
        _, top_indices = jt.topk(base_logits, k=1, dim=-1) # [Batch, 1]
        
        # [FIX] 之前这里多了一个 .unsqueeze(0)，导致变成了二维 [1, 1]
        # 我们只需要把它展平为一维 [Batch] (对于 Batch=1 就是 [1])
        candidates_logit = top_indices.view(-1) 
    else:
        if sampling == 'typical':
            candidates_logit = get_typical_one_token(base_logits, temperature, posterior_threshold, posterior_alpha).squeeze(0)
        elif sampling == 'nucleus':
            candidates_logit = get_nucleus_one_token(base_logits, temperature, top_p).squeeze(0)
        else:
            raise NotImplementedError
            
    # 2. Medusa TopK Candidates
    topk_vals, topk_indices = jt.topk(medusa_logits_last, TOPK, dim=-1)
    candidates_medusa_logits = topk_indices # [Heads, TopK]

    # 3. Combine (现在两个都是一维向量了，可以 concat)
    # candidates_logit: [1]
    # candidates_medusa_logits.view(-1): [N]
    candidates = jt.concat([candidates_logit, candidates_medusa_logits.view(-1)], dim=-1)

    # 4. Map to Tree
    # [FIX] 添加边界检查，确保 tree_indices 不超出 candidates 的范围
    # 这可以防止当 medusa_num_heads 与 medusa_choices 不匹配时出现索引越界
    max_candidate_idx = candidates.shape[0] - 1
    tree_indices_clamped = jt.clamp(tree_indices, 0, max_candidate_idx)
    tree_candidates = candidates[tree_indices_clamped]

    # 5. Extend and Retrieve Cartesian
    tree_candidates_ext = jt.concat([tree_candidates, jt.zeros((1,), dtype="int64")], dim=0)
    # [FIX] 添加边界检查，确保 retrieve_indices 不超出 tree_candidates_ext 的范围
    max_tree_idx = tree_candidates_ext.shape[0] - 1
    retrieve_indices_clamped = jt.clamp(retrieve_indices, 0, max_tree_idx)
    cart_candidates = tree_candidates_ext[retrieve_indices_clamped]
    
    # Unsqueeze for batch dim
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decoding Step.
    """
    position_ids = medusa_position_ids + input_ids.shape[1]

    # 使用 Jittor 原生操作进行边界检查（避免 .numpy() 的同步开销）
    vocab_size = model.config.vocab_size if hasattr(model, 'config') and hasattr(model.config, 'vocab_size') else 32000
    tree_candidates = jt.clamp(tree_candidates, 0, vocab_size - 1)
    
    tree_medusa_logits, outputs, tree_logits = model(
        input_ids=tree_candidates, # 注意参数名，这里传给 model.execute
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        medusa_forward=True,
    )
    
    # Reorder
    # logits = tree_logits[0, retrieve_indices]
    # retrieve_indices shape [Num_Candidates, Len]
    # tree_logits shape [Batch=1, Seq_Len_Tree, Vocab]
    
    # This advanced indexing [0, retrieve_indices] extracts logits for specific paths
    # Jittor indexing: 
    # tree_logits[0] -> [Seq_Len_Tree, Vocab]
    # retrieve_indices -> [N, L] (indices into Seq_Len_Tree)
    # result -> [N, L, Vocab]
    
    # 保存原始 logits（用于后续的 gather_and_reset 和采样）
    original_tree_logits = tree_logits
    original_tree_medusa_logits = tree_medusa_logits
    
    # [FIX] 添加边界检查，确保 retrieve_indices 不超出 tree_logits 的范围
    max_seq_idx = tree_logits.shape[1] - 1
    retrieve_indices_clamped = jt.clamp(retrieve_indices, 0, max_seq_idx)
    logits = tree_logits[0][retrieve_indices_clamped]
    
    # medusa_logits: [Heads, Batch, Seq, Vocab] -> [Heads, Seq, Vocab]
    medusa_logits = tree_medusa_logits[:, 0][:, retrieve_indices_clamped] # [Heads, N, L, Vocab]
    
    return medusa_logits, logits, original_tree_logits, original_tree_medusa_logits, outputs


def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha=0.09, 
    top_p=0.8, sampling='typical', fast=True, tokenizer=None, return_debug_info=False,
    medusa_choices=None, current_sequence=None
):
    # Greedy
    if temperature == 0:
        # [MODIFIED] Use topk instead of argmax
        _, pred_tokens = jt.topk(logits[:, :-1], k=1, dim=-1)
        pred_tokens = pred_tokens.squeeze(-1) # [Num_Cands, Len]
        
        gt_tokens = candidates[:, 1:]
        
        posterior_mask = (gt_tokens == pred_tokens).int()
        candidates_accept_length = jt.cumprod(posterior_mask, dim=1).sum(dim=1)

        # print("Debugging candidates_accept_length:")
        # print("Shape:", candidates_accept_length.shape)
        # print("Dtype:", candidates_accept_length.dtype)
        # print("Content:", candidates_accept_length)
        # # 确保张量中没有 NaN (Not a Number) 或 inf (infinity)
        # if candidates_accept_length.isnan().any() or candidates_accept_length.isinf().any():
        #     print("!!! WARNING: NaN or Inf detected in tensor !!!")

        
        accept_length = int(candidates_accept_length.max().item())
        
        if accept_length == 0:
            best_candidate = jt.array(0).int64()
        else:
            # best_candidate = jt.topk(candidates_accept_length).int64() # argmax here should be fine on 1D var
            best_candidate = jt.argmax(candidates_accept_length, dim=0)[0]
        
        # 准备调试信息（greedy模式）
        debug_info = None
        if return_debug_info:
            base_token = int(candidates[0, 0].item()) if candidates.shape[0] > 0 else None
            
            # 添加当前序列信息
            current_sequence_info = None
            if current_sequence is not None and tokenizer is not None:
                try:
                    # current_sequence是jt.Var，需要转换为numpy
                    if hasattr(current_sequence, 'numpy'):
                        seq_ids = current_sequence.numpy().tolist()
                    else:
                        seq_ids = current_sequence
                    
                    # 解码当前序列（显示最后20个token，避免太长）
                    if len(seq_ids) > 20:
                        seq_ids_display = seq_ids[-20:]
                        seq_text = tokenizer.decode(seq_ids_display, skip_special_tokens=False)
                        current_sequence_info = {
                            'token_ids': seq_ids,
                            'display_tokens': seq_ids_display,  # 最后20个token用于显示
                            'display_text': seq_text,
                            'total_length': len(seq_ids)
                        }
                    else:
                        seq_text = tokenizer.decode(seq_ids, skip_special_tokens=False)
                        current_sequence_info = {
                            'token_ids': seq_ids,
                            'display_tokens': seq_ids,
                            'display_text': seq_text,
                            'total_length': len(seq_ids)
                        }
                except Exception as e:
                    # 如果解码失败，至少保存token IDs
                    if hasattr(current_sequence, 'numpy'):
                        seq_ids = current_sequence.numpy().tolist()
                    else:
                        seq_ids = current_sequence
                    current_sequence_info = {
                        'token_ids': seq_ids,
                        'display_tokens': seq_ids[-20:] if len(seq_ids) > 20 else seq_ids,
                        'display_text': f"<解码失败: {str(e)}>",
                        'total_length': len(seq_ids)
                    }
            
            # 添加base model预测的token信息
            base_token_info = None
            if base_token is not None and tokenizer is not None:
                try:
                    base_token_text = tokenizer.decode([base_token], skip_special_tokens=False)
                    base_token_info = {
                        'token_id': base_token,
                        'token_text': base_token_text
                    }
                except:
                    base_token_info = {
                        'token_id': base_token,
                        'token_text': f"<token_{base_token}>"
                    }
            
            medusa_tree = []
            posterior_mask_np = posterior_mask.numpy()  # 转换为numpy数组
            for cand_idx in range(min(candidates.shape[0], 20)):
                path_tokens = candidates[cand_idx, :].numpy().tolist()
                if cand_idx < posterior_mask_np.shape[0]:
                    # posterior_mask是int类型（0或1），转换为bool
                    accept_status = [bool(x) for x in posterior_mask_np[cand_idx, :].tolist()]
                    full_accept_status = [True] + accept_status  # 第一个token（base model）总是被接受
                else:
                    full_accept_status = [True] * len(path_tokens)
                medusa_tree.append({
                    'path_idx': cand_idx,
                    'tokens': path_tokens,
                    'accept_status': full_accept_status,
                    'accept_length': int(candidates_accept_length[cand_idx].item()) if cand_idx < candidates_accept_length.shape[0] else 0
                })
            debug_info = {
                'base_token': base_token,
                'base_token_info': base_token_info,
                'current_sequence_info': current_sequence_info,
                'medusa_tree': medusa_tree,
                'best_candidate': int(best_candidate.item()),
                'accept_length': accept_length
            }
        
        if return_debug_info:
            return best_candidate, accept_length, debug_info
        else:
            return best_candidate, accept_length

    # Typical Acceptance 实现
    if sampling == 'typical':
        if fast:
            # Fast mode: 直接计算后验概率和阈值
            # logits shape: [Num_Cands, Max_Len, Vocab]
            # candidates shape: [Num_Cands, Max_Len]
            
            # 计算后验概率 (只考虑候选序列的位置，不包括最后一个)
            posterior_prob = nn.softmax(logits[:, :-1] / temperature, dim=-1)  # [Num_Cands, Max_Len-1, Vocab]
            
            # 获取候选token的概率
            # candidates[:, 1:] shape: [Num_Cands, Max_Len-1]
            candidates_idx = candidates[:, 1:].unsqueeze(-1)  # [Num_Cands, Max_Len-1, 1]
            
            # [FIX] 添加边界检查，确保 candidates_idx 中的 token ID 在有效范围内
            # 这可以防止 CUDA 非法地址访问错误
            vocab_size = posterior_prob.shape[-1]
            candidates_idx = jt.clamp(candidates_idx, 0, vocab_size - 1)
            
            candidates_prob = jt.gather(
                posterior_prob, dim=-1, index=candidates_idx
            ).squeeze(-1)  # [Num_Cands, Max_Len-1]
            
            # 计算熵: H = -sum(p * log(p))
            posterior_entropy = -jt.sum(
                posterior_prob * jt.log(posterior_prob + 1e-5), dim=-1
            )  # [Num_Cands, Max_Len-1]
            
            # 计算阈值: min(epsilon, delta * exp(-H))
            # posterior_threshold 对应 epsilon, posterior_alpha 对应 delta
            threshold = jt.minimum(
                jt.ones_like(posterior_entropy) * posterior_threshold,
                jt.exp(-posterior_entropy) * posterior_alpha,
            )  # [Num_Cands, Max_Len-1]
            
            # 判断哪些候选token被接受: p > threshold
            posterior_mask = candidates_prob > threshold  # [Num_Cands, Max_Len-1]
            
            # 处理EOS token：如果某个token是EOS且被接受，则后续token不应该被接受
            # 这是Typical Acceptance的正确逻辑：一旦遇到EOS，应该停止接受后续token
            # 这样确保选择的路径在EOS处停止，而不是继续接受EOS之后的token
            if tokenizer is not None and tokenizer.eos_token_id is not None:
                eos_token_id = tokenizer.eos_token_id
                # candidates[:, 1:] 是medusa预测的token（不包括第一个base model预测的token）
                candidates_tokens = candidates[:, 1:]  # [Num_Cands, Max_Len-1]
                
                # 找到EOS token的位置
                eos_mask = (candidates_tokens == eos_token_id)  # [Num_Cands, Max_Len-1]
                
                # 对于每条路径，如果EOS被接受，则后续token不应该被接受
                # 使用向量化操作：找到每条路径中第一个被接受的EOS的位置
                eos_and_accepted = eos_mask & posterior_mask  # [Num_Cands, Max_Len-1]
                
                # 对于每条路径，找到第一个被接受的EOS的位置
                # 使用cumsum找到第一个True的位置
                eos_and_accepted_int = eos_and_accepted.int()
                eos_cumsum = jt.cumsum(eos_and_accepted_int, dim=1)  # [Num_Cands, Max_Len-1]
                
                # 如果cumsum > 0，说明在这个位置或之前有EOS被接受
                # 对于EOS之后的位置，posterior_mask应该设为False
                has_eos_before = (eos_cumsum > 0)  # [Num_Cands, Max_Len-1]
                
                # 如果某个位置之前有EOS被接受，则这个位置不应该被接受
                # 但是，EOS本身应该被接受（如果它的概率满足条件）
                # 所以我们需要：如果某个位置之前（不包括自己）有EOS被接受，则这个位置不应该被接受
                # 使用shift操作：检查前一个位置是否有EOS被接受
                # 更简单：如果cumsum > 0 且当前位置不是EOS，则不应该被接受
                # 或者：如果cumsum > 0 且当前位置是EOS，则应该被接受（如果概率满足条件）
                # 实际上，我们需要：如果某个位置之前有EOS被接受，则这个位置不应该被接受
                
                # 使用更简单的方法：对于每条路径，找到第一个被接受的EOS的位置
                # 然后，从EOS之后的所有位置都不应该被接受
                # 由于Jittor的限制，我们使用numpy来处理
                eos_and_accepted_np = eos_and_accepted.numpy()
                posterior_mask_np = posterior_mask.numpy()
                
                for cand_idx in range(eos_and_accepted_np.shape[0]):
                    for pos_idx in range(eos_and_accepted_np.shape[1]):
                        # 如果这个位置是EOS且被接受
                        if eos_and_accepted_np[cand_idx, pos_idx]:
                            # 则后续所有位置都不应该被接受
                            posterior_mask_np[cand_idx, pos_idx+1:] = False
                            break  # 找到第一个EOS就停止
                
                # 转换回Jittor变量
                posterior_mask = jt.array(posterior_mask_np)
            
            # 计算每个候选的接受长度（连续接受的长度）
            # cumprod 计算累积乘积，如果中间有False，后续都是0
            posterior_mask_int = posterior_mask.int()
            candidates_accept_length = jt.cumprod(posterior_mask_int, dim=1).sum(dim=1)  # [Num_Cands]
            
            # 选择最佳候选
            accept_length = int(candidates_accept_length.max().item())
            
            if accept_length == 0:
                # 如果没有候选被接受，选择第一个
                best_candidate = jt.array(0).int64()
            else:
                # 找到所有达到最大接受长度的候选
                max_accept_length = candidates_accept_length.max()
                best_candidates_mask = (candidates_accept_length == max_accept_length)
                
                # 获取所有最佳候选的索引
                # 使用 numpy 转换来获取索引（Jittor 的 where 可能不支持这种用法）
                best_candidates_mask_np = best_candidates_mask.numpy()
                best_candidates_indices_np = np.where(best_candidates_mask_np)[0]
                
                if len(best_candidates_indices_np) == 0:
                    best_candidate = jt.array(0).int64()
                elif len(best_candidates_indices_np) == 1:
                    best_candidate = jt.array(best_candidates_indices_np[0]).int64()
                else:
                    # 如果有多个候选，根据似然选择最佳的一个
                    # 似然 = sum(log(p)) for accepted tokens
                    # 注意：需要处理 candidates_prob 可能为0的情况
                    best_candidates_indices = jt.array(best_candidates_indices_np).int64()
                    likelihood = jt.sum(
                        jt.log(candidates_prob[best_candidates_indices, :accept_length] + 1e-10), 
                        dim=-1
                    )  # [num_best_candidates]
                    best_idx = jt.argmax(likelihood, dim=0)[0]
                    best_candidate = best_candidates_indices[best_idx]
            
            # 准备调试信息
            debug_info = None
            if return_debug_info:
                # 获取base model预测的token（第一个token）
                base_token = int(candidates[0, 0].item()) if candidates.shape[0] > 0 else None
                
                # 添加当前序列信息
                current_sequence_info = None
                if current_sequence is not None and tokenizer is not None:
                    try:
                        # current_sequence是jt.Var，需要转换为numpy
                        if hasattr(current_sequence, 'numpy'):
                            seq_ids = current_sequence.numpy().tolist()
                        else:
                            seq_ids = current_sequence
                        
                        # 解码当前序列（显示最后20个token，避免太长）
                        if len(seq_ids) > 20:
                            seq_ids_display = seq_ids[-20:]
                            seq_text = tokenizer.decode(seq_ids_display, skip_special_tokens=False)
                            current_sequence_info = {
                                'token_ids': seq_ids,
                                'display_tokens': seq_ids_display,  # 最后20个token用于显示
                                'display_text': seq_text,
                                'total_length': len(seq_ids)
                            }
                        else:
                            seq_text = tokenizer.decode(seq_ids, skip_special_tokens=False)
                            current_sequence_info = {
                                'token_ids': seq_ids,
                                'display_tokens': seq_ids,
                                'display_text': seq_text,
                                'total_length': len(seq_ids)
                            }
                    except Exception as e:
                        # 如果解码失败，至少保存token IDs
                        if hasattr(current_sequence, 'numpy'):
                            seq_ids = current_sequence.numpy().tolist()
                        else:
                            seq_ids = current_sequence
                        current_sequence_info = {
                            'token_ids': seq_ids,
                            'display_tokens': seq_ids[-20:] if len(seq_ids) > 20 else seq_ids,
                            'display_text': f"<解码失败: {str(e)}>",
                            'total_length': len(seq_ids)
                        }
                
                # 添加base model预测的token信息
                base_token_info = None
                if base_token is not None and tokenizer is not None:
                    try:
                        base_token_text = tokenizer.decode([base_token], skip_special_tokens=False)
                        base_token_info = {
                            'token_id': base_token,
                            'token_text': base_token_text
                        }
                    except:
                        base_token_info = {
                            'token_id': base_token,
                            'token_text': f"<token_{base_token}>"
                        }
                
                # 获取medusa heads预测的token树
                # candidates shape: [Num_Cands, Max_Len]
                # 第一个token是base model预测的，后面的token是medusa预测的
                medusa_tree = []
                for cand_idx in range(min(candidates.shape[0], 20)):  # 限制显示前20条路径
                    path_tokens = candidates[cand_idx, :].numpy().tolist()
                    # 获取这条路径的接受情况
                    if cand_idx < posterior_mask.shape[0]:
                        accept_status = posterior_mask[cand_idx, :].numpy().tolist()
                        # 添加base token的接受状态（总是True）
                        full_accept_status = [True] + accept_status
                    else:
                        full_accept_status = [True] * len(path_tokens)
                    
                    medusa_tree.append({
                        'path_idx': cand_idx,
                        'tokens': path_tokens,
                        'accept_status': full_accept_status,
                        'accept_length': int(candidates_accept_length[cand_idx].item()) if cand_idx < candidates_accept_length.shape[0] else 0
                    })
                
                # 添加第一个medusa head的调试信息
                first_head_debug = None
                if medusa_choices is not None and len(medusa_choices) > 0:
                    # 找到第一个medusa head对应的路径
                    # 第一个medusa head对应medusa_choices中第一个元素（通常是[0]）
                    first_medusa_choice = medusa_choices[0]  # 通常是[0]
                    
                    # 在candidates中找到所有第一个位置是第一个medusa head预测的token的路径
                    # candidates[:, 1] 是所有候选路径中第一个medusa预测的token（索引1是第一个medusa位置）
                    # 从generate_candidates的逻辑来看：
                    # candidates = [base_token, medusa_head_0_token_0, medusa_head_0_token_1, ..., medusa_head_0_token_k-1, medusa_head_1_token_0, ...]
                    # 然后通过tree_indices和retrieve_indices映射到树结构
                    # 所以第一个medusa head的token在candidates中的原始位置是1到1+TOPK-1
                    # 但是，由于tree_indices和retrieve_indices的映射，在最终的cart_candidates中，位置可能不同
                    
                    # 获取所有候选路径中第一个medusa位置的token（索引1）
                    # 这些token就是第一个medusa head预测的token（虽然可能重复，因为不同的路径可能使用相同的token）
                    first_medusa_tokens = candidates[:, 1].numpy()  # [Num_Cands]
                    
                    # 找到所有不同的第一个medusa head的token
                    unique_first_medusa_tokens = list(set(first_medusa_tokens.tolist()))
                    unique_first_medusa_tokens = sorted(unique_first_medusa_tokens)[:10]  # 限制显示前10个
                    
                    # 对于每个第一个medusa head的token，找到对应的路径并计算p_original和阈值
                    first_head_tokens_info = []
                    for token_id in unique_first_medusa_tokens:
                        # 找到所有第一个位置是这个token的路径
                        paths_with_token = []
                        for cand_idx in range(candidates.shape[0]):
                            if candidates[cand_idx, 1].item() == token_id:
                                paths_with_token.append(cand_idx)
                        
                        if len(paths_with_token) > 0:
                            # 使用第一个路径来计算p_original和阈值
                            cand_idx = paths_with_token[0]
                            if cand_idx < candidates_prob.shape[0] and candidates_prob.shape[1] > 0:
                                # 第一个medusa head的token对应candidates_prob中的索引0（第一个medusa位置）
                                p_original = float(candidates_prob[cand_idx, 0].item())
                                threshold_val = float(threshold[cand_idx, 0].item())
                                is_accepted = bool(posterior_mask[cand_idx, 0].item()) if cand_idx < posterior_mask.shape[0] and posterior_mask.shape[1] > 0 else False
                                
                                first_head_tokens_info.append({
                                    'token_id': int(token_id),
                                    'p_original': p_original,
                                    'threshold': threshold_val,
                                    'is_accepted': is_accepted
                                })
                    
                    first_head_debug = {
                        'tokens': first_head_tokens_info
                    }
                
                debug_info = {
                    'base_token': base_token,
                    'base_token_info': base_token_info,
                    'current_sequence_info': current_sequence_info,
                    'medusa_tree': medusa_tree,
                    'best_candidate': int(best_candidate.item()),
                    'accept_length': accept_length,
                    'candidates_prob': candidates_prob.numpy().tolist() if 'candidates_prob' in locals() else None,
                    'threshold': threshold.numpy().tolist() if 'threshold' in locals() else None,
                    'first_head_debug': first_head_debug
                }
            
            if return_debug_info:
                return best_candidate, accept_length, debug_info
            else:
                return best_candidate, accept_length
        else:
            # Non-fast mode: 使用采样方式（暂时回退到fast模式）
            # TODO: 实现完整的 non-fast typical acceptance
            return evaluate_posterior(logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p, 'typical', True, tokenizer, return_debug_info)
    
    # Nucleus sampling (暂时回退到greedy)
    if sampling == 'nucleus':
        # TODO: 实现 nucleus sampling
        return evaluate_posterior(logits, candidates, 0, posterior_threshold, posterior_alpha, top_p, sampling, fast, tokenizer, return_debug_info)
    
    # 默认回退到greedy
    return evaluate_posterior(logits, candidates, 0, posterior_threshold, posterior_alpha, top_p, sampling, fast, tokenizer, return_debug_info)


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    tree_logits,           # 传入树计算出的 Logits [Batch, Tree_Size, Vocab]
    tree_medusa_logits,    # 传入树计算出的 Medusa Logits [Heads, Batch, Tree_Size, Vocab]
    tree_outputs,          # 传入 Tree Decoding 的 outputs（包含 hidden_states）
    new_token,
    past_key_values,       # 传入 KVCache 对象列表
    prev_len_before_tree,  # Tree Decoding 前的长度
):
    """
    更新推理状态：Gather KV Cache，提取最后一个被接受 token 的 hidden state。
    
    核心优化：
    1. 从 Tree Decoding 产生的 KV Cache 中 Gather 被接受的路径
    2. 从 tree_outputs 中提取最后一个被接受 token 的 hidden state，用于下一步的 medusa heads 预测
    3. 不再需要 Correction Step Forward，因为 hidden state 已经在 tree decoding 中计算好了
    """
    # 1. 解析被接受的路径
    bc_idx = best_candidate.item()
    al_val = int(accept_length)
    
    # [CRITICAL] Gather 所有被接受的节点，包括 Index 0 (Base Model Prediction)
    # retrieve_indices[bc_idx] 包含了 [0, 1, 4...] 这样的展平索引
    # 我们需要把这些位置的 KV 全部整理到 Cache 的末尾
    accepted_tree_indices = retrieve_indices[bc_idx, :al_val + 1]
    
    # 转换为 Jittor Var
    if not isinstance(accepted_tree_indices, jt.Var):
        accepted_tree_indices = jt.array(accepted_tree_indices, dtype="int64")
    else:
        accepted_tree_indices = accepted_tree_indices.int64()
    
    # 确保 prev_len 是 int
    if hasattr(prev_len_before_tree, "item"):
        prev_len_int = int(prev_len_before_tree.item())
    else:
        prev_len_int = int(prev_len_before_tree)
    
    # 2. 整理 KV Cache (Gather & Reset)
    # 无论 al_val 是多少，我们都执行 gather。
    # 如果 al_val=0，我们 gather [0] 到 prev_len。这等价于确认第一个 token 并重置长度。
    for layer_caches in past_key_values:
        for cache in layer_caches:  # K and V
            cache.gather_and_reset(accepted_tree_indices, prev_len_int)
    
    # 3. 更新 input_ids (历史记录)
    # candidates: [Num_Candidates, Max_Len]
    accepted_tokens = candidates[bc_idx:bc_idx+1, :al_val + 1]  # [1, al_val+1]
    input_ids = jt.concat([input_ids, accepted_tokens], dim=-1)
    
    # 4. 提取最后一个被接受 token 的 hidden state
    # tree_outputs.hidden_states[-1] shape: [Batch, Tree_Size, Hidden_Size]
    # 我们需要找到最后一个被接受节点在 tree 中的索引
    final_accepted_node_index = retrieve_indices[bc_idx, al_val].item()
    
    # [FIX] 添加边界检查，防止索引超出范围
    tree_hidden_states = tree_outputs.hidden_states[-1]  # [Batch, Tree_Size, Hidden_Size]
    tree_size = tree_hidden_states.shape[1]
    if final_accepted_node_index >= tree_size:
        final_accepted_node_index = tree_size - 1
    elif final_accepted_node_index < 0:
        final_accepted_node_index = 0
    
    # 提取最后一个被接受 token 的 hidden state: [Batch, 1, Hidden_Size]
    last_hidden_state = tree_hidden_states[:, final_accepted_node_index:final_accepted_node_index+1, :]
    # 确保 hidden state 是 float32 类型，避免类型不匹配
    last_hidden_state = last_hidden_state.float32()
    
    # 5. 提取最后一个被接受 token 的 logits（用于下一步的 medusa heads）
    # tree_logits shape: [Batch, Tree_Size, Vocab]
    final_logits = tree_logits[:, final_accepted_node_index:final_accepted_node_index+1, :]  # [1, 1, Vocab]
    
    # [FIX] 检查 final_logits 是否为空
    if final_logits.shape[1] == 0:
        final_logits = tree_logits[:, -1:, :]  # [1, 1, Vocab]
    
    new_token += al_val + 1
    
    return input_ids, last_hidden_state, final_logits, new_token