import jittor as jt
from jittor import nn

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
        
        # Suppress BOS token (usually token 1) and EOS token (usually token 2)
        # Also suppress PAD token (usually token 0) if any
        special_tokens = [0, 1, 2]  # <unk>, <s>, </s>
        neg_inf = jt.float32(-1e9)  # Use large negative instead of -inf for numerical stability
        
        for tok_id in special_tokens:
            base_logits[:, tok_id] = neg_inf
            medusa_logits_last[:, tok_id] = neg_inf
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
    tree_candidates = candidates[tree_indices]

    # 5. Extend and Retrieve Cartesian
    tree_candidates_ext = jt.concat([tree_candidates, jt.zeros((1,), dtype="int64")], dim=0)
    cart_candidates = tree_candidates_ext[retrieve_indices]
    
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
    
    logits = tree_logits[0][retrieve_indices]
    
    # medusa_logits: [Heads, Batch, Seq, Vocab] -> [Heads, Seq, Vocab]
    medusa_logits = tree_medusa_logits[:, 0][:, retrieve_indices] # [Heads, N, L, Vocab]
    
    return medusa_logits, logits, outputs


def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha=0.09, 
    top_p=0.8, sampling='typical', fast=True
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
            
        return best_candidate, accept_length

    # Recursive call for other samplings (Placeholder for now)
    return evaluate_posterior(logits, candidates, 0, posterior_threshold, posterior_alpha, top_p, sampling, fast)


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update state.
    """
    prev_input_len = input_ids.shape[1]
    
    # 1. Map Best Candidate -> Tree Indices -> Original Indices
    # retrieve_indices: [Num_Cands, Max_Len]
    # best_candidate is scalar index
    
    # Python int conversion for slicing
    bc_idx = best_candidate.item()
    al_val = int(accept_length) # passed as int/scalar
    
    # select_indices = retrieve_indices[bc_idx, :al_val + 1] + prev_input_len
    select_indices = retrieve_indices[bc_idx, :al_val + 1] + prev_input_len
    
    # 2. Append tokens to input_ids
    # candidates shape: [Num_Cands, Max_Len]
    # We need to select candidates[bc_idx, :al_val+1] and add batch dim
    new_tokens = candidates[bc_idx:bc_idx+1, :al_val + 1]  # [1, al_val+1]
    input_ids = jt.concat([input_ids, new_tokens], dim=-1)
    
    # 3. Update KV Cache (The trickiest part)
    # past_key_values_data: [Layers*2, Batch, Heads, MaxPos, Dim]
    # We want to perform:
    # dst = data[..., prev_len : prev_len + len, :]
    # src = data[..., select_indices, :]
    # dst[:] = src
    
    # 获取源数据 (Advanced Indexing on dim -2)
    # select_indices is [Len]
    # Jittor indexing: data[..., select_indices, :]
    tgt = past_key_values_data[..., select_indices, :]
    
    # 写入目标位置
    # In Jittor, we construct slices
    # Shape of data: [L*2, B, H, MaxPos, D]
    # Dim -2 is MaxPos
    
    target_start = prev_input_len
    target_end = prev_input_len + tgt.shape[-2]
    
    # construct slice tuple: (:, :, :, start:end, :)
    slices = [slice(None)] * len(past_key_values_data.shape)
    slices[-2] = slice(target_start, target_end)
    
    past_key_values_data[tuple(slices)] = tgt
    
    # 4. Update Lengths
    # 注意：Jittor 中 assign() 对整个数组可能不work，需要逐元素更新
    new_length = prev_input_len + tgt.shape[-2]
    for idx in range(current_length_data.shape[0]):
        current_length_data[idx] = new_length
    
    # 5. Update Logits (Return logits for the next step)
    # logits shape: [Num_Cands, Max_Len, Vocab]
    # We need logits[bc_idx, al_val:al_val+1, :] -> [1, Vocab] -> [1, 1, Vocab]
    # medusa_logits shape: [Heads, Num_Cands, Max_Len, Vocab]
    # We need medusa_logits[:, bc_idx, al_val:al_val+1, :] -> [Heads, 1, Vocab] -> [Heads, 1, 1, Vocab]
    
    logits = logits[bc_idx:bc_idx+1, al_val:al_val+1, :]  # [1, 1, Vocab]
    medusa_logits = medusa_logits[:, bc_idx:bc_idx+1, al_val:al_val+1, :]  # [Heads, 1, 1, Vocab]
    
    new_token += al_val + 1
    
    return input_ids, logits, medusa_logits, new_token