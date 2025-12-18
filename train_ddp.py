import os
import time

# --- [Hack] 解决 PyTorch 和 Jittor 的库冲突 ---
# 在导入 torch 之前隐藏 GPU，防止加载 CUDA 库
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import torch
from transformers import AutoTokenizer
# 恢复 GPU 可见性
del os.environ["CUDA_VISIBLE_DEVICES"]

# --- Jittor 导入 ---
import jittor as jt
from jittor import nn, optim
jt.flags.use_cuda = 1 # 显式开启

import argparse
import json
import gc
import shutil
from datasets import Dataset, load_from_disk

# 导入模型
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig

# --- 简单的文件锁同步工具 ---
def wait_for_rank0(run_dir, rank):
    """
    强制非Rank 0进程等待，直到Rank 0完成数据处理。
    """
    flag_file = os.path.join(run_dir, "rank0_data_done.flag")
    
    if rank == 0:
        # Rank 0: 处理完数据后创建文件
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
        with open(flag_file, 'w') as f:
            f.write("done")
        print(f"[Sync] Rank 0 created completion flag at {flag_file}")
    else:
        # Rank N: 轮询检查文件是否存在
        print(f"[Sync] Rank {rank} waiting for data processing...")
        while not os.path.exists(flag_file):
            time.sleep(5) # 每5秒检查一次
        print(f"[Sync] Rank {rank} detected completion flag. Proceeding.")

# --- Jittor-兼容的数据集类 ---
class JittorTextDataset(jt.dataset.Dataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.hf_dataset = hf_dataset 
        # 只设置 total_len，其他属性手动设置
        self.set_attrs(total_len=len(self.hf_dataset))

    def __getitem__(self, index):
        data = self.hf_dataset[index]['input_ids']
        return jt.array(data).int64()

def main(args):
    # 开启显存管理 (通过环境变量设置更稳妥，这里作为备用)
    if hasattr(jt.flags, 'use_cuda_managed_memory'):
        jt.flags.use_cuda_managed_memory = 1 

    if jt.in_mpi:
        rank = jt.rank
        world_size = jt.world_size
        print(f"[Info] Rank {rank}/{world_size} initialized.")
    else:
        rank = 0
        world_size = 1
        print("[Info] Single GPU mode.")

    # --- Step 1: 加载基座模型 ---
    if rank == 0:
        print("--- Step 1: Loading Pre-trained Base Model ---")
    
    with open(os.path.join(args.base_model_path, "config.json"), 'r') as f:
        llama_config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(llama_config_dict)

    base_model = LlamaForCausalLM(llama_config)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    else:
        # 手动设置 flag
        base_model.model.gradient_checkpointing = True
    
    # if rank == 0:
    #     print(f"Loading converted Jittor weights from: {args.jittor_weights_path}")
    
    # # 转换 FP16
    # raw_weights = jt.load(args.jittor_weights_path)
    # fp16_weights = {}
    # for k, v in raw_weights.items():
    #     fp16_weights[k] = v.astype("float16")

    # base_model.load_parameters(fp16_weights)
    # del fp16_weights
    # jt.gc()

    print("Casting base model structure to FP16...")
    base_model.float16() 
    
    if rank == 0:
        print(f"Loading converted Jittor weights from: {args.jittor_weights_path}")
    
    # 2. 加载 FP16 权重
    raw_weights = jt.load(args.jittor_weights_path)
    base_model.load_parameters(raw_weights)
    
    # 3. 强力 GC
    del raw_weights
    import gc
    gc.collect()
    jt.gc()
    
    # 4. 打印当前显存占用 (Debug)
    if rank == 0:
        jt.display_memory_info()
        print("Base model loaded and memory optimized.")

    # --- Step 2: 冻结基座 ---
    for param in base_model.parameters():
        param.stop_grad()

    # --- Step 3: 组装 Medusa ---
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_heads,
        medusa_num_layers=1,
        hidden_size=llama_config.hidden_size,
        vocab_size=llama_config.vocab_size
    )
    
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train()
    
    # --- Step 4: 数据准备 (修复版同步机制) ---
    if rank == 0:
        print("\n--- Step 4: Preparing Dataset ---")
    
    cache_path = os.path.join(args.output_dir, "processed_dataset_cache")
    # 清理旧的完成标记，防止逻辑错误
    flag_file = os.path.join(args.output_dir, "rank0_data_done.flag")
    if rank == 0 and os.path.exists(flag_file):
        os.remove(flag_file)
    
    # === Rank 0 处理数据 ===
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False, legacy=True)
        
        if os.path.exists(cache_path) and not args.overwrite_cache:
            print(f"Loading cached dataset from {cache_path}...")
            tokenized_datasets = load_from_disk(cache_path)
        else:
            print(f"Processing raw data from {args.local_dataset_path}...")
            # ... (数据加载逻辑保持不变) ...
            try:
                with open(args.local_dataset_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                processed_texts = []
                for item in raw_data:
                    if "conversations" in item:
                        text_content = " ".join([conv["value"] for conv in item["conversations"]])
                        processed_texts.append({"text": text_content})
                    elif "text" in item:
                        processed_texts.append({"text": item["text"]})
                
                raw_datasets = Dataset.from_dict({"text": [item["text"] for item in processed_texts]})
                
                def tokenize_function(examples):
                    texts = [str(t) for t in examples["text"]]
                    tokenized = tokenizer(texts)
                    concatenated_examples = {k: sum(tokenized[k], []) for k in tokenized.keys()}
                    total_length = len(concatenated_examples[list(tokenized.keys())[0]])
                    total_length = (total_length // args.seq_len) * args.seq_len
                    result = {
                        k: [t[i : i + args.seq_len] for i in range(0, total_length, args.seq_len)]
                        for k, t in concatenated_examples.items()
                    }
                    result["labels"] = result["input_ids"].copy()
                    return result

                print("Tokenizing and chunking (this may take a while)...")
                tokenized_datasets = raw_datasets.map(
                    tokenize_function, 
                    batched=True, 
                    remove_columns=["text"],
                    num_proc=8 # 增加进程数
                )
                
                print(f"Saving dataset to cache: {cache_path}")
                tokenized_datasets.save_to_disk(cache_path)
                
            except Exception as e:
                print(f"Error processing dataset: {e}")
                exit(1)

    # === [FIX] 强同步机制 ===
    # Rank 0 创建文件，Rank 1 轮询等待
    wait_for_rank0(args.output_dir, rank)
    
    # 确保 Jittor 层面也同步一下
    if jt.in_mpi:
        jt.sync_all()

    # === 所有 Rank 加载数据 ===
    tokenized_datasets = load_from_disk(cache_path)
    train_dataset = JittorTextDataset(tokenized_datasets)
    
    # === [FIX] 手动设置分布式属性 (绕过 set_attrs 断言) ===
    # 直接赋值，更加稳健
    train_dataset.part_id = rank
    train_dataset.part_count = world_size
    
    # 设置常规属性
    dataloader = train_dataset.set_attrs(
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    if rank == 0:
        print(f"Dataset ready. Total samples: {len(train_dataset)}. Distributed over {world_size} GPUs.")

    # --- Step 5: 训练循环 ---
    optimizer = optim.AdamW(model.medusa_head.parameters(), lr=args.lr)
    medusa_decay = 0.8
    
    print(f"Rank {rank} start training loop...")    
    
    for epoch in range(args.epochs):
        for batch_idx, input_ids in enumerate(dataloader):
            # [Memory Opt] 激进 GC: 每一步都尝试清理，防止显存碎片
            # if batch_idx % 50 == 0:
            #    jt.gc()

            medusa_logits = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=False 
            )
            
            total_loss = jt.zeros(1)
            for k in range(args.medusa_heads):
                head_logits = medusa_logits[k]
                offset = k + 1
                if offset >= input_ids.shape[1]: break
                
                pred = head_logits[:, :-offset, :]
                target = input_ids[:, offset:]
                
                loss_k = nn.cross_entropy_loss(pred.reshape(-1, llama_config.vocab_size), target.reshape(-1))
                total_loss += loss_k * (medusa_decay ** k)
            
            optimizer.step(total_loss)
            
            # 只让 Rank 0 打印日志
            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {batch_idx} | Loss: {total_loss.item():.4f}")

    # --- Step 6: 保存 (仅 Rank 0) ---
    if rank == 0:
        print("\n--- Step 6: Saving Trained Medusa Heads ---")
        save_path = os.path.join(args.output_dir, "medusa_lm_head.jtr")
        jt.save(model.medusa_head.parameters(), save_path)
        print(f"Medusa heads saved to: {save_path}")
        
        # 删除同步标志文件
        if os.path.exists(flag_file):
            os.remove(flag_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--jittor_weights_path", type=str, required=True)
    parser.add_argument("--local_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./medusa_checkpoints_ddp")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2) # 分布式下每张卡的batch size
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--medusa_heads", type=int, default=5)
    parser.add_argument("--overwrite_cache", action="store_true")

    args = parser.parse_args()
    main(args)