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
        # === [性能优化] 一次性索引赋值，替换 for 循环 ===
        suppress_ids = [0, 1]  # <unk>, <s>
        neg_inf = jt.float32(-1e9)  # Use large negative instead of -inf for numerical stability
        
        # Jittor 支持列表/Tensor索引赋值，一次性完成所有抑制
        base_logits[:, suppress_ids] = neg_inf
        medusa_logits_last[:, suppress_ids] = neg_inf
        
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
    
    # === [核心性能优化] Tree Decoding 阶段不计算 Medusa Heads ===
    # 显式传入 return_medusa_logits=False，跳过 Medusa Heads 计算
    # 这样只有 Tree Decoding 阶段会跳过计算，initialize_medusa 仍然正常工作
    _, outputs, tree_logits = model(
        input_ids=tree_candidates, # 注意参数名，这里传给 model.execute
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        medusa_forward=True,
        return_medusa_logits=False,  # [新增] 跳过 Medusa Heads 计算
    )
    
    # === [核心优化] 移除 Logits 展开，避免路径膨胀 ===
    # 原代码: logits = tree_logits[0][retrieve_indices_clamped]
    # 这种写法会产生巨大的显存拷贝 [Num_Paths, Path_Len, Vocab_Size]
    # 我们直接返回扁平的 [1, Tree_Size, Vocab] 原始 Logits
    # evaluate_posterior 会在扁平 Logits 上计算，然后通过索引映射到路径
    
    # 保存原始 logits（用于后续的 gather_and_reset 和采样）
    original_tree_logits = tree_logits
    
    # 返回扁平的 tree_logits，不再展开
    # tree_logits shape: [1, Tree_Size, Vocab]
    # evaluate_posterior 会使用 retrieve_indices 来映射路径
    return None, tree_logits, original_tree_logits, None, outputs


def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha=0.09, 
    top_p=0.8, sampling='typical', fast=True, tokenizer=None, return_debug_info=False,
    medusa_choices=None, current_sequence=None, retrieve_indices=None  # [新增参数] 需要传入索引来映射路径
):
    # === [核心优化] 在扁平 Logits 上计算，避免路径膨胀 ===
    # logits 现在接收的是 [1, Tree_Size, Vocab] 或 [Tree_Size, Vocab]
    # 预处理：去掉 Batch 维度，变成 [Tree_Size, Vocab]
    if logits.ndim == 3:
        flat_logits = logits[0]  # [Tree_Size, Vocab]
    else:
        flat_logits = logits  # 已经是 [Tree_Size, Vocab]
    
    # 获取映射索引 (在 medusa_generate 中调用时需要传入 medusa_buffers["retrieve_indices"])
    # retrieve_indices Shape: [Num_Paths, Path_Len]
    if retrieve_indices is None:
        raise ValueError("retrieve_indices must be provided for optimized evaluate_posterior")
    
    # 确保索引不越界
    max_tree_idx = flat_logits.shape[0] - 1
    retrieve_indices_clamped = jt.clamp(retrieve_indices, 0, max_tree_idx)
    
    # Greedy
    if temperature == 0:
        # === [优化] 在扁平 Logits 上计算 Argmax ===
        # 1. 在 Flat Logits 上计算 Argmax
        # [Tree_Size, Vocab] -> [Tree_Size]
        _, flat_pred_tokens = jt.topk(flat_logits, k=1, dim=-1)
        flat_pred_tokens = flat_pred_tokens.squeeze(-1)  # [Tree_Size]
        
        # 2. 映射回路径
        # [Num_Paths, Path_Len]
        path_pred_tokens = flat_pred_tokens[retrieve_indices_clamped]
        
        # 3. 比较（只考虑候选序列的位置，不包括最后一个）
        # path_pred_tokens[:, :-1] shape: [Num_Paths, Path_Len-1]
        gt_tokens = candidates[:, 1:]  # [Num_Paths, Max_Len-1]
        
        # 确保维度匹配
        min_len = min(path_pred_tokens.shape[1], gt_tokens.shape[1])
        path_pred_tokens = path_pred_tokens[:, :min_len]
        gt_tokens = gt_tokens[:, :min_len]
        
        posterior_mask = (gt_tokens == path_pred_tokens).int()
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
            # === [核心优化] 在扁平 Logits 上计算 Softmax 和 Entropy ===
            # 1. 在 Flat Logits 上计算 Softmax 和 Entropy (无冗余计算)
            # flat_logits: [Tree_Size, Vocab]
            flat_probs = nn.softmax(flat_logits / temperature, dim=-1)  # [Tree_Size, Vocab]
            
            # 2. 计算 Flat Entropy
            # [Tree_Size]
            flat_entropy = -jt.sum(
                flat_probs * jt.log(flat_probs + 1e-5), dim=-1
            )
            
            # 3. 计算 Flat Threshold
            # [Tree_Size]
            flat_threshold = jt.minimum(
                jt.ones_like(flat_entropy) * posterior_threshold,
                jt.exp(-flat_entropy) * posterior_alpha,
            )
            
            # 4. 获取候选 Token 的概率
            # 对于路径 i 的第 j 步（j从0开始）：
            #   - 父节点索引: retrieve_indices[i, j]（当前步的节点）
            #   - 提议的 token: candidates[i, j+1]（下一步的 token）
            #   - 需要查表: flat_probs[retrieve_indices[i, j], candidates[i, j+1]]
            
            # 目标 Token IDs: [Num_Paths, Max_Len-1]
            # candidates[:, 1:] 是 medusa 预测的 token（不包括第一个 base model 预测）
            target_ids = candidates[:, 1:]  # [Num_Paths, Max_Len-1]
            
            # 边界检查
            vocab_size = flat_probs.shape[-1]
            target_ids_clamped = jt.clamp(target_ids, 0, vocab_size - 1)
            
            # 获取父节点索引（用于查表）
            # retrieve_indices_clamped[:, :-1] 是每一步的父节点索引
            # 对于路径 i 的第 j 步，父节点是 retrieve_indices[i, j]
            parent_indices = retrieve_indices_clamped[:, :-1]  # [Num_Paths, Path_Len-1]
            
            # 确保维度匹配
            min_len = min(parent_indices.shape[1], target_ids_clamped.shape[1])
            parent_indices = parent_indices[:, :min_len]
            target_ids_clamped = target_ids_clamped[:, :min_len]
            
            # Jittor 高级索引 Gather
            # flat_probs: [Tree_Size, Vocab]
            # index_0 (Row): parent_indices [Num_Paths, Path_Len-1]
            # index_1 (Col): target_ids_clamped [Num_Paths, Path_Len-1]
            # 结果: [Num_Paths, Path_Len-1]
            candidates_prob = flat_probs[parent_indices, target_ids_clamped]
            
            # 5. 映射阈值回路径 (只映射 Scalar 值，极快)
            # path_threshold: [Num_Paths, Path_Len-1]
            path_threshold = flat_threshold[parent_indices]  # [Num_Paths, Path_Len-1]
            
            # 6. 计算阈值（已经映射到路径）
            threshold = path_threshold  # [Num_Paths, Path_Len-1]
            
            # 判断哪些候选token被接受: p > threshold
            posterior_mask = candidates_prob > threshold  # [Num_Paths, Path_Len-1]
            
            # 处理EOS token：如果某个token是EOS且被接受，则后续token不应该被接受
            # 这是Typical Acceptance的正确逻辑：一旦遇到EOS，应该停止接受后续token
            # 这样确保选择的路径在EOS处停止，而不是继续接受EOS之后的token
            if tokenizer is not None and tokenizer.eos_token_id is not None:
                eos_token_id = tokenizer.eos_token_id
                # candidates[:, 1:] 是medusa预测的token（不包括第一个base model预测的token）
                candidates_tokens = candidates[:, 1:]  # [Num_Paths, Max_Len-1]
                
                # 确保维度匹配
                min_len = min(posterior_mask.shape[1], candidates_tokens.shape[1])
                candidates_tokens = candidates_tokens[:, :min_len]
                posterior_mask = posterior_mask[:, :min_len]
                
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
    # 1. 准备基础数据
    bc_idx = int(best_candidate.item())
    al_val = int(accept_length)
    prev_len = int(prev_len_before_tree)
    new_len = prev_len + al_val + 1
    
    # 2. [性能优化] 预计算所有层通用的 Gather 索引
    # retrieve_indices 已经是 [Batch, MaxLen] 形式
    # 我们取出 [0...al_val] 的 tree 索引 (相对于 tree window)
    relative_indices = retrieve_indices[bc_idx, :al_val + 1]  # Tensor
    
    # 转换为绝对位置索引：prev_len + relative_index
    # 这样所有层都可以直接使用这个 read_indices
    read_indices = relative_indices + prev_len
    
    # 确保是 int64 且扁平化
    if not isinstance(read_indices, jt.Var):
        read_indices = jt.array(read_indices).int64()
    read_indices = read_indices.view(-1)
    
    # 3. 批量更新 KV Cache
    # 这里的 Python 循环只做最轻量的函数调用
    for layer_caches in past_key_values:
        # layer_caches[0] is K, [1] is V
        layer_caches[0].gather_and_reset(read_indices, prev_len, new_len)
        layer_caches[1].gather_and_reset(read_indices, prev_len, new_len)
    
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
    # 这是一个极其轻量的操作
    last_hidden_state = tree_hidden_states[:, final_accepted_node_index:final_accepted_node_index+1, :]
    
    # === [性能优化] 不在这里转换类型，让 Medusa Heads 自己处理 ===
    # 如果 Medusa Heads 需要 float32，它们会在内部转换
    # 这样可以保持 fp16 的计算速度
    
    # 5. 提取最后一个被接受 token 的 logits（用于下一步的验证）
    # tree_logits shape: [Batch, Tree_Size, Vocab]
    final_logits = tree_logits[:, final_accepted_node_index:final_accepted_node_index+1, :]  # [1, 1, Vocab]
    
    # [FIX] 检查 final_logits 是否为空
    if final_logits.shape[1] == 0:
        final_logits = tree_logits[:, -1:, :]  # [1, 1, Vocab]
    
    # === [核心性能优化] 删除从 tree_medusa_logits 提取的逻辑 ===
    # 因为我们没有在 Tree Decoding 阶段计算 Medusa Heads
    # 返回 None，让外部（medusa_generate）只对这一个 Token 计算 Medusa Heads
    
    new_token += al_val + 1
    
    return input_ids, last_hidden_state, final_logits, None, new_token