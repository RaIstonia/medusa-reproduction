import os
import time
import math
import shutil
import copy
import numpy as np
from functools import partial
from tqdm import tqdm

# --- [Hack] 解决 PyTorch 和 Jittor 的库冲突 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import torch
from transformers import AutoTokenizer
del os.environ["CUDA_VISIBLE_DEVICES"]

# --- Jittor 导入 ---
import jittor as jt
from jittor import nn, optim
jt.flags.use_cuda = 1 

import argparse
import json
import gc
from datasets import Dataset, load_from_disk

from medusa.model_moe.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model_moe.modeling_llama import LlamaForCausalLM, LlamaConfig

# ... (辅助函数 save_checkpoint, wait_for_rank0, process_vicuna_batch, JittorInstructDataset 保持不变) ...
# 为了节省篇幅，这里复用您原有的辅助函数代码
# ...
def save_checkpoint(model, output_dir, tag="latest", metric=None):
    save_path = os.path.join(output_dir, f"checkpoint-{tag}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if jt.in_mpi:
        jt.sync_all()
        
    weights_path = os.path.join(save_path, "medusa_lm_head.jtr")
    jt.save(model.medusa_blocks.state_dict(), weights_path)
    
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model.config.to_dict(), f, indent=4)
    
    if metric is not None:
        with open(os.path.join(save_path, "loss.txt"), "w") as f:
            f.write(str(metric))
            
    print(f"Saved {tag} checkpoint to: {save_path}")

def wait_for_rank0(run_dir, rank):
    flag_file = os.path.join(run_dir, "rank0_data_done.flag")
    if rank == 0:
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
        with open(flag_file, 'w') as f:
            f.write("done")
    else:
        while not os.path.exists(flag_file):
            time.sleep(5)

IGNORE_TOKEN_ID = -100
# ... (process_vicuna_batch, JittorInstructDataset 复用之前代码) ...
def process_vicuna_batch(examples, tokenizer_path, max_len):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, legacy=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id 
    separator = " "
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    system_ids = tokenizer(system_prompt + separator, add_special_tokens=False).input_ids
    batch_input_ids = []
    batch_labels = []
    for conversations in examples['conversations']:
        input_ids = copy.deepcopy(system_ids)
        labels = [IGNORE_TOKEN_ID] * len(input_ids)
        for turn in conversations:
            role = turn["from"]
            content = turn["value"]
            if role == "human":
                role_str = "USER: "
                prefix_ids = tokenizer(role_str, add_special_tokens=False).input_ids
                content_ids = tokenizer(content, add_special_tokens=False).input_ids
                ids = prefix_ids + content_ids + tokenizer(" ", add_special_tokens=False).input_ids
                input_ids += ids
                labels += [IGNORE_TOKEN_ID] * len(ids)
            elif role == "gpt":
                role_str = "ASSISTANT: "
                prefix_ids = tokenizer(role_str, add_special_tokens=False).input_ids
                content_ids = tokenizer(content, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
                input_ids += prefix_ids
                labels += [IGNORE_TOKEN_ID] * len(prefix_ids)
                input_ids += content_ids
                labels += content_ids
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
        else:
            pad_len = max_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [IGNORE_TOKEN_ID] * pad_len
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
    return {"input_ids": batch_input_ids, "labels": batch_labels}

class JittorInstructDataset(jt.dataset.Dataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.hf_dataset = hf_dataset 
        self.set_attrs(total_len=len(self.hf_dataset))
    def __getitem__(self, index):
        item = self.hf_dataset[index]
        return jt.array(item['input_ids']).int64(), jt.array(item['labels']).int64()

def get_lr(current_step, total_steps, warmup_steps, learning_rate, min_lr=0.0):
    if current_step < warmup_steps:
        return float(learning_rate * (current_step / max(1, warmup_steps)))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(min_lr, 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress)))

# [新增] 计算负载均衡 Loss
def compute_load_balancing_loss(router_logits_list, num_experts, top_k=1):
    """
    Load Balancing Loss = num_experts * sum(fraction_selected * mean_probs)
    """
    total_aux_loss = jt.zeros(1)
    if not router_logits_list:
        return total_aux_loss

    for router_logits in router_logits_list:
        # router_logits: [batch, seq_len, num_experts]
        logits_flat = router_logits.view(-1, num_experts)
        probs = jt.nn.softmax(logits_flat, dim=-1)
        
        # 1. 平均概率 P_i (Differentiable)
        mean_probs = probs.mean(dim=0) # [num_experts]
        
        # 2. 选中频率 f_i (Non-differentiable, need stop_grad)
        _, selected_indices = jt.topk(probs, top_k, dim=-1) # [N, k]
        
        # 将选中索引转为 one-hot 统计
        # Jittor 没有直接的 topk indices 转 onehot mean 的函数，手动实现
        # 创建一个全零矩阵 [N, num_experts]
        # 这种 Scatter 操作在 Python 层略慢，但在训练中占比不大
        
        # 简易实现：使用 Mask
        mask = jt.zeros((logits_flat.shape[0], num_experts))
        for i in range(num_experts):
            # 检查是否在 TopK 中
            mask[:, i] = (selected_indices == i).any(dim=-1).float32()
            
        fraction_selected = mask.mean(dim=0)
        fraction_selected = fraction_selected.stop_grad() # 关键
        
        # 3. 计算 Loss
        aux_loss = (mean_probs * fraction_selected).sum() * num_experts
        total_aux_loss += aux_loss
        
    return total_aux_loss / len(router_logits_list)

def main(args):
    # Paths setup
    args.base_model_path = os.path.abspath(args.base_model_path)
    args.jittor_weights_path = os.path.abspath(args.jittor_weights_path)
    args.local_dataset_path = os.path.abspath(args.local_dataset_path)
    args.output_dir = os.path.abspath(args.output_dir)

    if hasattr(jt.flags, 'use_cuda_managed_memory'):
        jt.flags.use_cuda_managed_memory = 1 
    
    # ... (MPI setup, Model Loading 同之前) ...
    if jt.in_mpi:
        rank = jt.rank
        world_size = jt.world_size
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        print(f"Working Directory: {os.getcwd()}")
        print(f"Output Directory: {args.output_dir}")

    # --- Step 1: Load Base Model ---
    if rank == 0:
        print("--- Step 1: Loading Base Model (FP16) ---")
    
    with open(os.path.join(args.base_model_path, "config.json"), 'r') as f:
        llama_config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(llama_config_dict)

    base_model = LlamaForCausalLM(llama_config)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    else:
        base_model.model.gradient_checkpointing = True
    base_model.float16()
    
    raw_weights = jt.load(args.jittor_weights_path)
    processed_weights = {k: jt.array(v).float16() for k, v in raw_weights.items()}
    base_model.load_parameters(processed_weights)
    del raw_weights, processed_weights
    jt.gc()

    print(f"Freezing base model parameters... (LoRA: {args.enable_lora_training})")
    for name, param in base_model.named_parameters():
        if args.enable_lora_training and "lora" in name:
            pass 
        else:
            param.stop_grad()

    # --- Step 2: Medusa MoE ---
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_heads,
        medusa_num_layers=args.medusa_num_layers,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size,
        enable_lora_training=args.enable_lora_training,
        moe_num_experts=args.moe_num_experts, # [新增]
        moe_top_k=args.moe_top_k              # [新增]
    )
    
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train()
    
    # ... (Dataset loading 同之前) ...
    cache_path = os.path.join(args.output_dir, "processed_vicuna_dataset_cache")
    flag_file = os.path.join(args.output_dir, "rank0_data_done.flag")
    if rank == 0:
        if os.path.exists(flag_file): os.remove(flag_file)
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        if os.path.exists(cache_path) and not args.overwrite_cache:
            print(f"Loading cached dataset from {cache_path}...")
        else:
            # ... process dataset ...
            # (省略重复代码，与之前一致)
            try:
                with open(args.local_dataset_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                hf_dataset = Dataset.from_list(raw_data)
                process_func = partial(process_vicuna_batch, tokenizer_path=args.base_model_path, max_len=args.seq_len)
                tokenized_dataset = hf_dataset.map(process_func, batched=True, num_proc=args.num_proc, remove_columns=hf_dataset.column_names, desc="Tokenizing")
                tokenized_dataset.save_to_disk(cache_path)
            except Exception as e:
                print(e); exit(1)
                
    wait_for_rank0(args.output_dir, rank)
    if jt.in_mpi: jt.sync_all()
    tokenized_datasets = load_from_disk(cache_path)
    train_dataset = JittorInstructDataset(tokenized_datasets)
    train_dataset.part_id = rank
    train_dataset.part_count = world_size
    dataloader = train_dataset.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=min(8, args.num_proc // 2))

    # --- Step 4: Optimizer ---
    params_to_optimize = list(model.medusa_blocks.parameters())
    if args.enable_lora_training:
        lora_params = [p for n, p in model.base_model.named_parameters() if "lora" in n]
        params_to_optimize.extend(lora_params)
        
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    global_step = 0
    current_iter = 0
    best_loss = float('inf')
    medusa_decay = 0.8
    
    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        accum_loss_var = jt.zeros(1)
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            current_iter += 1
            current_lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # [修改] 调用 execute 时请求 router logits
            medusa_logits, router_logits_list = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=False,
                return_router_logits=True
            )
            
            # 1. Main Loss (Next Token Prediction)
            main_loss = jt.zeros(1)
            for k in range(args.medusa_heads):
                head_logits = medusa_logits[k]
                offset = k + 1
                if offset >= input_ids.shape[1]: break
                pred_logits = head_logits[:, :-offset, :]
                target_ids = labels[:, offset:]
                pred_flat = pred_logits.reshape(-1, llama_config.vocab_size).float32()
                target_flat = target_ids.reshape(-1)
                valid_mask = (target_flat != IGNORE_TOKEN_ID).float32()
                loss_all = nn.cross_entropy_loss(pred_flat, target_flat, reduction='none')
                valid_loss = loss_all * valid_mask
                num_valid = valid_mask.sum().stop_grad()
                loss_k = valid_loss.sum() / (num_valid + 1e-6)
                main_loss += loss_k * (medusa_decay ** k)
            
            # 2. Aux Loss (Load Balancing)
            aux_loss = jt.zeros(1)
            if len(router_logits_list) > 0:
                aux_loss = compute_load_balancing_loss(router_logits_list, args.moe_num_experts, args.moe_top_k)
            
            # Total Loss
            total_loss = main_loss + args.aux_loss_weight * aux_loss
            
            loss_scaled = total_loss / args.gradient_accumulation_steps
            optimizer.backward(loss_scaled)
            accum_loss_var += total_loss
            
            if current_iter % args.gradient_accumulation_steps == 0:
                optimizer.clip_grad_norm(1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if rank == 0:
                    progress_bar.update(1)
                    if global_step % args.logging_steps == 0:
                        avg_loss = accum_loss_var.item() / args.gradient_accumulation_steps
                        # 打印 aux loss 用于监控
                        cur_aux = aux_loss.item()
                        progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "Aux": f"{cur_aux:.4f}", "LR": f"{current_lr:.6f}"})
                
                # Save Logic
                if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    curr_loss = accum_loss_var.item() / args.gradient_accumulation_steps
                    save_checkpoint(model, args.output_dir, tag="latest")
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        save_checkpoint(model, args.output_dir, tag="best", metric=best_loss)

                accum_loss_var = jt.zeros(1)
                if global_step % 50 == 0: jt.gc()

    if jt.in_mpi: jt.sync_all()
    if rank == 0:
        progress_bar.close()
        save_checkpoint(model, args.output_dir, tag="final")
    if jt.in_mpi: jt.sync_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (原有 args) ...
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--jittor_weights_path", type=str, required=True)
    parser.add_argument("--local_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="medusa_checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--medusa_num_layers", type=int, default=1)
    parser.add_argument("--medusa_heads", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--enable_lora_training", action="store_true")
    parser.add_argument("--num_proc", type=int, default=16)

    # MoE Args
    parser.add_argument("--moe_num_experts", type=int, default=4, help="Number of experts in MoE block")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-k experts to activate")
    parser.add_argument("--aux_loss_weight", type=float, default=0.01, help="Weight for load balancing loss")
    
    args = parser.parse_args()
    main(args)