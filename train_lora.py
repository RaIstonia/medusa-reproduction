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

# 导入模型定义
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
# 确保 llama_model 中包含 LoRA 相关的类和函数
from medusa.model.modeling_llama import (
    LlamaForCausalLM, 
    LlamaConfig
)

from medusa.lora.monkey_patching import LoRAConfig, inject_lora, mark_only_lora_as_trainable

# ===========================
# 1. 辅助函数与保存逻辑 (更新版)
# ===========================

def save_checkpoint(model, output_dir, tag="latest", metric=None, save_lora=False):
    """
    保存检查点: 
    1. Medusa Head 权重
    2. Config
    3. (可选) LoRA 权重
    """
    save_path = os.path.join(output_dir, f"checkpoint-{tag}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if jt.in_mpi:
        jt.sync_all()
        
    # 1. 保存 Medusa Head
    weights_path = os.path.join(save_path, "medusa_lm_head.jtr")
    jt.save(model.medusa_head.state_dict(), weights_path)
    
    # 2. (新增) 保存 LoRA 权重
    if save_lora:
        lora_weights = {}
        # 遍历 Base Model 提取 LoRA 参数
        for name, param in model.base_model.state_dict().items():
            if "lora_" in name:
                lora_weights[name] = param
        
        lora_save_path = os.path.join(save_path, "lora_weights.jtr")
        jt.save(lora_weights, lora_save_path)
        print(f"Saved LoRA weights to {lora_save_path} ({len(lora_weights)} keys)")
    
    # 3. 保存 Config
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
# 2. Loss 计算类 (新增)
# ===========================

class MedusaLoss(nn.Module):
    def __init__(self, medusa_num_heads, medusa_decay_coefficient=0.8, vocab_size=32000):
        super().__init__()
        self.medusa_num_heads = medusa_num_heads
        self.medusa_decay_coefficient = medusa_decay_coefficient
        self.vocab_size = vocab_size
        self.ignore_index = -100

    def execute(self, medusa_logits, base_logits, input_ids, labels, enable_lora=False):
        """
        Args:
            medusa_logits: [num_heads, batch, seq_len, vocab]
            base_logits: [batch, seq_len, vocab] (如果是 LoRA 模式则不为 None)
            labels: [batch, seq_len]
        """
        total_loss = jt.zeros(1)
        medusa_loss_sum = jt.zeros(1)
        
        # 1. Base Model Loss (Next Token Prediction) - 仅在 LoRA 模式下计算
        if enable_lora and base_logits is not None:
            # [关键修复] Causal LM 需要 Shift！
            # 在 Causal LM 中：
            # - 输入序列: [A, B, C]
            # - Base Model 在位置 i 的 logits 预测的是位置 i+1 的 token
            # - 所以 base_logits[0] 预测 labels[1], base_logits[1] 预测 labels[2], 等等
            # - Logits 去掉最后一个，Labels 去掉第一个
            shift_logits = base_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            
            # 转为 float32 防止溢出
            shift_logits = shift_logits.float32()
            
            base_loss = nn.cross_entropy_loss(
                shift_logits.reshape(-1, self.vocab_size), 
                shift_labels.reshape(-1),
                ignore_index=self.ignore_index
            )
            total_loss += base_loss
        else:
            base_loss = jt.zeros(1)

        
        for k in range(self.medusa_num_heads):
            head_logits = medusa_logits[k] # [B, L, V]
            
            # [关键修复] k=0 -> offset=2 (预测 t+2), k=1 -> offset=3 (预测 t+3)
            offset = k + 2
            
            # 边界检查
            if offset >= labels.shape[1]: 
                break
            pred_logits = head_logits[:, :-offset, :]
            
            # Labels 切片: 去掉前面 offset 个
            # 对应上面的 Target: Label 2, Label 3, Label 4
            target_ids = labels[:, offset:]
            
            pred_flat = pred_logits.reshape(-1, self.vocab_size)
            target_flat = target_ids.reshape(-1)
            
            loss_k = nn.cross_entropy_loss(
                pred_flat, 
                target_flat, 
                ignore_index=self.ignore_index
            )
            
            # 加权累加
            decay_weight = self.medusa_decay_coefficient ** k
            medusa_loss_sum += loss_k * decay_weight

        total_loss += medusa_loss_sum
        return total_loss, base_loss, medusa_loss_sum

# ===========================
# 3. 数据集处理 (保持不变)
# ===========================

IGNORE_TOKEN_ID = -100

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

# ===========================
# 4. 训练主逻辑 (大幅更新)
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

    # --- Step 1: 加载 Base Model ---
    if rank == 0:
        print("--- Step 1: Loading Base Model ---")
    
    with open(os.path.join(args.base_model_path, "config.json"), 'r') as f:
        llama_config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(llama_config_dict)

    base_model = LlamaForCausalLM(llama_config)
    
    # [LoRA] 如果启用 LoRA，在加载权重前注入 (或者加载后注入也可以，通常加载后注入更安全)
    # 这里我们采用：先加载 Pretrained Weights -> 注入 LoRA -> 处理梯度
    
    # [1.1] 加载原始权重
    base_model.float16() # 使用 FP16
    if rank == 0:
        print(f"Loading base weights from: {args.jittor_weights_path}")
    raw_weights = jt.load(args.jittor_weights_path)
    processed_weights = {k: jt.array(v).float16() for k, v in raw_weights.items()}
    base_model.load_parameters(processed_weights)
    del raw_weights, processed_weights
    jt.gc()

    # [1.2] LoRA 注入
    if args.enable_lora_training:
        if rank == 0:
            print(f"--- Injecting LoRA (Rank={args.lora_rank}) ---")
        lora_config = LoRAConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"] # 可以根据需要添加 k_proj, o_proj 等
        )
        base_model = inject_lora(base_model, lora_config)
        
        # [关键] 设置梯度：Base冻结，LoRA解冻
        mark_only_lora_as_trainable(base_model)
    else:
        # 非 LoRA 模式：全量冻结 Base Model
        for param in base_model.parameters():
            param.stop_grad()

    # --- Step 2: 组装 Medusa ---
    if rank == 0:
        print("--- Step 2: Initializing Medusa Head ---")
        
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_heads,
        medusa_num_layers=args.medusa_num_layers,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size,
        enable_lora_training=args.enable_lora_training # 传递配置
    )
    
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train()
    
    # [关键] 确保 Medusa Head 是可训练的
    for param in model.medusa_head.parameters():
        param.start_grad()

    # 打印参数统计
    if rank == 0:
        trainable_params = 0
        all_params = 0
        for param in model.parameters():
            all_params += param.numel()
            if not param.is_stop_grad(): # Jittor 中 is_stop_grad() 为 False 表示需要梯度
                trainable_params += param.numel()
        print(f"Total Params: {all_params}")
        print(f"Trainable Params: {trainable_params} ({(trainable_params/all_params)*100:.4f}%)")

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
        num_workers=args.num_proc // 2
    )

    # --- Step 4: 训练配置 ---
    # [关键] 收集所有可训练参数 (LoRA + Medusa)
    trainable_parameters = []
    for param in model.parameters():
        if not param.is_stop_grad():
            trainable_parameters.append(param)

    optimizer = optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    # 实例化 Loss
    criterion = MedusaLoss(
        medusa_num_heads=args.medusa_heads, 
        medusa_decay_coefficient=args.medusa_decay,
        vocab_size=llama_config.vocab_size
    )
    
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 增加参数：max_grad_norm
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    
    if rank == 0:
        print(f"Total Steps: {total_steps}, Warmup: {warmup_steps}")
        print(f"Global Batch Size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Max Grad Norm: {max_grad_norm}")

    global_step = 0
    current_iter = 0
    best_loss = float('inf')
    
    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        accum_loss_var = jt.zeros(1)
        accum_base_loss = jt.zeros(1)
        
        # 增加一个 flag 标记本次 step 是否包含 nan
        step_has_nan = False
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            current_iter += 1
            
            # LR Scheduler
            current_lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Forward
            # 如果开启 LoRA，我们需要 output_orig=True 来获取 Base Model 的 Logits 以计算 Base Loss
            returns = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=args.enable_lora_training 
            )
            
            if args.enable_lora_training:
                medusa_logits, _, base_logits = returns
            else:
                medusa_logits = returns
                base_logits = None # Medusa-only 模式下不需要 base logits
            
            # Loss Calculation
            total_loss, base_loss_val, medusa_loss_val = criterion(
                medusa_logits, 
                base_logits, 
                input_ids, 
                labels, 
                enable_lora=args.enable_lora_training
            )

            # --- [关键修复 3] NaN 检测 ---
            # 立即检测 Loss 是否为 NaN 或 Inf
            # 使用 .item() 会触发同步，虽然慢一点点但能保命
            try:
                loss_val_check = total_loss.item()
                if math.isnan(loss_val_check) or math.isinf(loss_val_check):
                    if rank == 0:
                        print(f"[Warning] Loss is {loss_val_check} at step {global_step}, skipping batch.")
                    step_has_nan = True
                    # 这里为了防止 Jittor 显存不释放，可以手动清理一下图（可选）
                    total_loss = None 
                    jt.gc()
                    continue # 跳过本次 forward 的 backward
            except Exception as e:
                if rank == 0:
                    print(f"[Warning] Error checking loss value: {e}, skipping batch.")
                step_has_nan = True
                jt.gc()
                continue

            # Backward
            loss_scaled = total_loss / args.gradient_accumulation_steps
            optimizer.backward(loss_scaled)
            
            accum_loss_var += total_loss
            accum_base_loss += base_loss_val
            
            # Step
            if current_iter % args.gradient_accumulation_steps == 0:
                # --- [关键修复 4] 安全的 Optimizer Step ---
                if not step_has_nan:
                    # 梯度裁剪
                    optimizer.clip_grad_norm(max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                else:
                    if rank == 0:
                        print(f"[Info] Skipping optimizer step due to NaN in accumulation window.")
                    
                    # 1. 清空梯度
                    optimizer.zero_grad()
                    
                    # 2. [关键修复] 强制清理显存！
                    # 如果不加这行，坏掉的计算图会残留在显存里，导致下一轮 OOM
                    jt.gc()
                    
                    step_has_nan = False # 重置标记
                
                if rank == 0:
                    progress_bar.update(1)
                
                # Logging
                if rank == 0 and global_step % args.logging_steps == 0:
                    avg_loss = accum_loss_var.item() / args.gradient_accumulation_steps
                    avg_base = accum_base_loss.item() / args.gradient_accumulation_steps
                    
                    desc_str = f"L:{avg_loss:.3f}"
                    if args.enable_lora_training:
                        desc_str += f" (B:{avg_base:.3f})"
                    desc_str += f" LR:{current_lr:.5f}"
                    
                    progress_bar.set_postfix_str(desc_str)
                
                # Checkpointing
                if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    curr_loss = accum_loss_var.item() / args.gradient_accumulation_steps
                    # 保存 (包括 LoRA)
                    save_checkpoint(model, args.output_dir, tag="latest", save_lora=args.enable_lora_training)
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        save_checkpoint(model, args.output_dir, tag="best", metric=best_loss, save_lora=args.enable_lora_training)

                accum_loss_var = jt.zeros(1)
                accum_base_loss = jt.zeros(1)
                
                if global_step % 50 == 0:
                    jt.gc()

    if jt.in_mpi:
        jt.sync_all()

    if rank == 0:
        progress_bar.close()
        print("\n--- Saving Final Model ---")
        try:
            save_checkpoint(model, args.output_dir, tag="final", save_lora=args.enable_lora_training)
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
    parser.add_argument("--base_model_path", type=str, required=True, help="HuggingFace model path (for tokenizer & config)")
    parser.add_argument("--jittor_weights_path", type=str, required=True, help="Converted Jittor weights file")
    parser.add_argument("--local_dataset_path", type=str, required=True, help="Path to Vicuna format JSON")
    parser.add_argument("--output_dir", type=str, default="medusa_checkpoints")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--medusa_num_layers", type=int, default=1)
    parser.add_argument("--medusa_heads", type=int, default=3)
    parser.add_argument("--medusa_decay", type=float, default=0.8)
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--overwrite_cache", action="store_true")
    
    # LoRA Training Params
    parser.add_argument("--enable_lora_training", action="store_true", 
                        help="Enable LoRA training (Unfreeze LoRA params & calculate Base Model loss)")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Data Processing
    parser.add_argument("--num_proc", type=int, default=16, help="Num processes for data map")
    
    args = parser.parse_args()
    main(args)