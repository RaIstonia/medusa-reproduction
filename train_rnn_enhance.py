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

# 导入模型
from medusa.model_rnn_enhance.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model_rnn_enhance.modeling_llama import LlamaForCausalLM, LlamaConfig

# ===========================
# 1. 辅助函数与保存逻辑
# ===========================

def save_checkpoint(model, output_dir, tag="latest", metric=None):
    """保存检查点: 只保存 Head 权重和 Config"""
    save_path = os.path.join(output_dir, f"checkpoint-{tag}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if jt.in_mpi:
        jt.sync_all()
        
    weights_path = os.path.join(save_path, "medusa_lm_head.jtr")
    # [修改点 1] medusa_head -> medusa_blocks
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

# ===========================
# 2. 数据集处理 (Vicuna 格式)
# ===========================

IGNORE_TOKEN_ID = -100

def process_vicuna_batch(examples, tokenizer_path, max_len):
    # 在 worker 进程中重新导入，避免 fork 问题
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

# ===========================
# 3. 训练主逻辑
# ===========================

def get_lr(current_step, total_steps, warmup_steps, learning_rate, min_lr=0.0):
    if current_step < warmup_steps:
        return float(learning_rate * (current_step / max(1, warmup_steps)))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(min_lr, 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress)))

def main(args):
    # 路径绝对化
    args.base_model_path = os.path.abspath(args.base_model_path)
    args.jittor_weights_path = os.path.abspath(args.jittor_weights_path)
    args.local_dataset_path = os.path.abspath(args.local_dataset_path)
    args.output_dir = os.path.abspath(args.output_dir)

    if hasattr(jt.flags, 'use_cuda_managed_memory'):
        jt.flags.use_cuda_managed_memory = 1 

    if jt.in_mpi:
        rank = jt.rank
        world_size = jt.world_size
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        print(f"Working Directory: {os.getcwd()}")
        print(f"Output Directory: {args.output_dir}")

    # --- Step 1: 加载模型 (FP16) ---
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

    # [PRECISION] Base Model 使用 FP16 以节省显存
    base_model.float16()
    
    if rank == 0:
        print(f"Loading weights: {args.jittor_weights_path}")
    
    raw_weights = jt.load(args.jittor_weights_path)
    # [PRECISION] 转换权重为 FP16
    processed_weights = {k: jt.array(v).float16() for k, v in raw_weights.items()}
    base_model.load_parameters(processed_weights)
    
    del raw_weights, processed_weights
    import gc
    gc.collect()
    jt.gc()

    # [修改点 2] 智能冻结参数
    # 如果开启 LoRA，只冻结非 LoRA 参数；否则冻结所有 Base Model 参数
    print(f"Freezing base model parameters... (LoRA Training: {args.enable_lora_training})")
    for name, param in base_model.named_parameters():
        if args.enable_lora_training and "lora" in name:
            # 如果是 LoRA 层，不调用 stop_grad (保持可训练)
            pass 
        else:
            param.stop_grad()

    # --- Step 2: 组装 Medusa ---
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_heads,
        medusa_num_layers=args.medusa_num_layers,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size,
        enable_lora_training=args.enable_lora_training
    )
    
    model = MedusaModel(medusa_config, base_model=base_model)
    # [PRECISION] Head 保持 float32，避免梯度精度问题
    model.train()
    
    # --- Step 3: 数据准备 ---
    cache_path = os.path.join(args.output_dir, "processed_vicuna_dataset_cache")
    flag_file = os.path.join(args.output_dir, "rank0_data_done.flag")
    
    if rank == 0:
        if os.path.exists(flag_file):
            os.remove(flag_file)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
    
    if rank == 0:
        if os.path.exists(cache_path) and not args.overwrite_cache:
            print(f"Loading cached dataset from {cache_path}...")
        else:
            print(f"Start processing data (Num Proc: {args.num_proc})...")
            try:
                with open(args.local_dataset_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                hf_dataset = Dataset.from_list(raw_data)
                
                process_func = partial(
                    process_vicuna_batch, 
                    tokenizer_path=args.base_model_path, 
                    max_len=args.seq_len
                )
                tokenized_dataset = hf_dataset.map(
                    process_func,
                    batched=True,
                    num_proc=args.num_proc,
                    remove_columns=hf_dataset.column_names,
                    desc="Tokenizing"
                )
                print(f"Saving dataset to disk: {cache_path}")
                tokenized_dataset.save_to_disk(cache_path)
            except Exception as e:
                print(f"Data Processing Error: {e}")
                exit(1)

    wait_for_rank0(args.output_dir, rank)
    if jt.in_mpi: jt.sync_all()

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache path {cache_path} does not exist!")

    tokenized_datasets = load_from_disk(cache_path)
    train_dataset = JittorInstructDataset(tokenized_datasets)
    train_dataset.part_id = rank
    train_dataset.part_count = world_size
    
    dataloader = train_dataset.set_attrs(
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(8, args.num_proc // 2)
    )

    # --- Step 4: 训练配置 ---
    
    # [修改点 3] 收集需要优化的参数
    params_to_optimize = list(model.medusa_blocks.parameters()) # medusa_head -> medusa_blocks
    
    if args.enable_lora_training:
        print("Adding LoRA parameters to optimizer...")
        lora_params = [p for n, p in model.base_model.named_parameters() if "lora" in n]
        params_to_optimize.extend(lora_params)
        print(f"Found {len(lora_params)} LoRA parameters.")
        
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    
    medusa_decay = 0.8
    
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if rank == 0:
        print(f"Total Steps: {total_steps}, Warmup: {warmup_steps}")
        print(f"Global Batch Size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    global_step = 0
    current_iter = 0
    best_loss = float('inf')
    
    model.train()
    
    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # [OPT] 只保留 Loss 累积 (Jittor Var)
        accum_loss_var = jt.zeros(1)
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            current_iter += 1
            
            # LR Scheduler
            current_lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # === [修正] Hydra Teacher Forcing 数据准备 ===
            prev_token_embeddings = []
            if model.embed_tokens is not None:
                batch_size, seq_len = labels.shape
                
                # Head 1 需要 Head 0 的 Target (Offset 1)
                # Head 2 需要 Head 1 的 Target (Offset 2)
                # 所以只需要准备 Head 1 到 Head N-1 的输入
                for k in range(args.medusa_heads - 1):  # 只需要准备 Head 1 到 Head N-1
                    target_offset = k + 1
                    
                    if target_offset < seq_len:
                        # 1. 取出对应的 Label (向左 Shift)
                        # Head k+1 的输入对应 labels 的位置 target_offset...N
                        target_ids_slice = labels[:, target_offset:]  # [Batch, Seq-Offset]
                        
                        # 2. 处理 IGNORE_INDEX (-100) -> 0 (避免 Embedding 层报错)
                        valid_mask = (target_ids_slice != IGNORE_TOKEN_ID)
                        safe_ids = jt.where(valid_mask, target_ids_slice, jt.zeros_like(target_ids_slice))
                        
                        # 3. 获取 Embedding
                        embeds = model.embed_tokens(safe_ids)  # [Batch, Seq-Offset, Dim] (这是 float16)
                        
                        # === [修改] 强制转换为 float32 以匹配 Medusa Head 的权重 ===
                        embeds = embeds.float32()
                        # =======================================================
                        
                        # 4. 填充回原始长度 (为了和 hidden_state 形状匹配)
                        # hidden_state 是 [Batch, Seq, Dim]
                        # 我们在右侧填充 0 (因为后面算 Loss 时 mask 也会把右侧 mask 掉)
                        pad_len = seq_len - embeds.shape[1]
                        if pad_len > 0:
                            # 获取 embed_dim
                            embed_dim = embeds.shape[-1]
                            # padding 也必须是 float32
                            padding = jt.zeros((batch_size, pad_len, embed_dim), dtype="float32")
                            embeds_padded = jt.concat([embeds, padding], dim=1)
                        else:
                            embeds_padded = embeds
                        
                        prev_token_embeddings.append(embeds_padded)
                    else:
                        # 极端情况：offset 超过序列长度，append 全 0
                        # 获取 embed_dim (从第一个 block 或直接使用 hidden_size)
                        if hasattr(model.medusa_blocks[0], 'embed_dim'):
                            embed_dim = model.medusa_blocks[0].embed_dim
                        elif model.embed_tokens is not None and hasattr(model.embed_tokens, 'weight'):
                            embed_dim = model.embed_tokens.weight.shape[1]
                        else:
                            embed_dim = model.hidden_size
                        # 这里的 dummy 也必须是 float32
                        dummy = jt.zeros((batch_size, seq_len, embed_dim), dtype="float32")
                        prev_token_embeddings.append(dummy)
            
            # Forward
            # medusa_forward=True 触发 MedusaModel.execute 中的串行 Block 逻辑
            # prev_token_embeddings 用于 Teacher Forcing
            medusa_logits = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=False,
                prev_token_embeddings=prev_token_embeddings  # [Hydra] Teacher Forcing
            )
            
            total_loss = jt.zeros(1)
            
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
                
                # stop_grad on denominator for stability
                num_valid = valid_mask.sum().stop_grad()
                # 避免除以 0
                loss_k = valid_loss.sum() / jt.maximum(num_valid, 1.0)
                
                total_loss += loss_k * (medusa_decay ** k)

            # Backward
            loss_scaled = total_loss / args.gradient_accumulation_steps
            optimizer.backward(loss_scaled)
            
            accum_loss_var += total_loss
            
            # Step
            if current_iter % args.gradient_accumulation_steps == 0:
                optimizer.clip_grad_norm(1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if rank == 0:
                    progress_bar.update(1)
                
                # Logging (Lazy Sync)
                if rank == 0 and global_step % args.logging_steps == 0:
                    avg_loss = accum_loss_var.item() / args.gradient_accumulation_steps
                    progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.6f}"})
                
                # Save Logic
                if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    curr_loss = accum_loss_var.item() / args.gradient_accumulation_steps
                    save_checkpoint(model, args.output_dir, tag="latest")
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        save_checkpoint(model, args.output_dir, tag="best", metric=best_loss)

                accum_loss_var = jt.zeros(1)
                
                if global_step % 50 == 0:
                    jt.gc()

    # 在训练循环结束后
    if jt.in_mpi:
        jt.sync_all()  # 确保所有进程完成训练

    if rank == 0:
        progress_bar.close()
        print("\n--- Saving Final Model ---")
        try:
            save_checkpoint(model, args.output_dir, tag="final")
            print("Final model saved successfully")
        except Exception as e:
            print(f"Error saving final model: {e}")
        finally:
            if os.path.exists(flag_file):
                os.remove(flag_file)

    if jt.in_mpi:
        jt.sync_all()
        print(f"Rank {rank}: Process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--jittor_weights_path", type=str, required=True)
    parser.add_argument("--local_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="medusa_checkpoints")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=1)
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
    
    # LoRA Training
    parser.add_argument("--enable_lora_training", action="store_true", 
                        help="Enable LoRA training mode. When enabled, gradients will flow back to Base Model's LoRA layers.")
    
    # Data Processing
    parser.add_argument("--num_proc", type=int, default=16, help="Num processes for data map")
    
    args = parser.parse_args()
    main(args)