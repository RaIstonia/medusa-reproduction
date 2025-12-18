# 文件名: train_medusa_distributed.py

import jittor as jt
from jittor import nn, optim
import os
import argparse
import json
from transformers import AutoTokenizer
from datasets import Dataset # 我们仍然使用 Dataset 对象来利用其强大的 .map() 功能

# 导入您自己的模型定义
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig

# --- Jittor-兼容的数据集类 ---
class JittorTextDataset(jt.dataset.Dataset):
    """一个简单的数据集包装器，用于处理来自 Hugging Face datasets 的 tokenized 数据"""
    def __init__(self, tokenized_data):
        super().__init__()
        self.input_ids = tokenized_data['input_ids']
        self.set_attrs(total_len=len(self.input_ids))

    def __getitem__(self, index):
        return jt.array(self.input_ids[index]).int64()

def main(args):
    """主训练函数"""
    # --- [DDP 修改 1] 初始化分布式环境 ---
    # Jittor 会自动从环境变量（由 mpirun 设置）中获取 rank 和 world_size
    jt.init(distributed=True)
    rank = jt.rank
    world_size = jt.world_size
    
    if rank == 0:
        print(f"Jittor DDP initialized. Total GPUs: {world_size}")

    # --- 1. 加载预训练的基座模型 ---
    if rank == 0: print("--- Step 1: Loading Pre-trained Base Model ---")
    with open(os.path.join(args.base_model_path, "config.json"), 'r') as f:
        llama_config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(llama_config_dict)
    base_model = LlamaForCausalLM(llama_config)
    
    if rank == 0: print(f"Loading converted Jittor weights from: {args.jittor_weights_path}")
    base_model.load_parameters(jt.load(args.jittor_weights_path))
    if rank == 0: print("Base model weights loaded successfully.")

    # --- 2. 冻结基座模型的所有参数 ---
    if rank == 0: print("\n--- Step 2: Freezing Base Model Parameters ---")
    for param in base_model.parameters():
        param.stop_grad()

    # --- 3. 组装完整的 Medusa 模型 ---
    if rank == 0: print("\n--- Step 3: Assembling Medusa Model ---")
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_heads, medusa_num_layers=1,
        hidden_size=llama_config.hidden_size, vocab_size=llama_config.vocab_size
    )
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train()
    if rank == 0: print("Medusa model assembled.")
    
    # --- 4. 准备真实文本数据集 (本地加载 + DDP 优化) ---
    if rank == 0: print("\n--- Step 4: Preparing Dataset ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    
    # --- [本地加载 修改] ---
    # 只有主进程 (rank 0) 执行文件加载和预处理
    tokenized_datasets = None
    if rank == 0:
        print(f"Rank 0: Loading local dataset from: {args.local_dataset_path}")
        try:
            with open(args.local_dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 从 ShareGPT 格式中提取文本
            processed_texts = []
            for item in raw_data:
                if "conversations" in item:
                    text_content = " ".join([conv["value"] for conv in item["conversations"] if "value" in conv])
                    if text_content: # 确保不添加空字符串
                        processed_texts.append({"text": text_content})
            
            if not processed_texts:
                raise ValueError("No valid text data extracted from 'conversations' key.")
            
            # 转换为 Hugging Face Dataset 对象以使用 .map()
            raw_datasets = Dataset.from_list(processed_texts)
            # 为了让所有进程都有 'train' 这个key，我们在这里处理
            # 实际应用中可以不分割，直接使用整个数据集
            raw_datasets = raw_datasets.train_test_split(test_size=0.01, seed=42)
            
            print(f"Local dataset loaded and processed by Rank 0. Total items: {len(raw_datasets)}")

            def tokenize_function(examples):
                # 强制转换为字符串以处理潜在的 None 或非字符串值
                examples["text"] = [str(t) for t in examples["text"]]
                tokenized = tokenizer(examples["text"])
                concatenated_examples = {k: sum(tokenized[k], []) for k in tokenized.keys()}
                total_length = len(concatenated_examples[list(tokenized.keys())[0]])
                total_length = (total_length // args.seq_len) * args.seq_len
                
                result = {
                    k: [t[i : i + args.seq_len] for i in range(0, total_length, args.seq_len)]
                    for k, t in concatenated_examples.items()
                }
                return result

            print("Rank 0: Tokenizing and chunking dataset...")
            tokenized_datasets = raw_datasets.map(
                tokenize_function, batched=True, remove_columns=["text"]
            )
            print("Rank 0: Dataset preparation complete.")
            
        except Exception as e:
            print(f"FATAL: An error occurred on Rank 0 during data preparation: {e}")
            # 在分布式环境中，一个进程失败，最好让所有进程都退出
            tokenized_datasets = None # 设为None以触发同步后的退出

    # --- [DDP 修改 2] 同步点 ---
    # Jittor DDP 中，需要确保 rank 0 完成了任务，其他 rank 才能继续。
    # 一个简单的方法是使用 jt.sync_all()。
    # 更稳妥的方法是，Rank 0 将处理好的数据保存到磁盘，其他 Rank 再从磁盘加载。
    # 这里我们使用 datasets 库的缓存机制，它能自动实现类似效果。
    # 我们只需让所有进程都调用 .map()，非 Rank 0 进程会直接命中 Rank 0 创建的缓存。
    
    # 为了简单且有效，让所有进程都构建对象，但只有 rank 0 真正执行计算
    if rank != 0:
        # 非主进程需要等待主进程完成缓存，然后它们会直接加载缓存
        # 我们重复上面的逻辑，但由于缓存的存在，速度会非常快
        with open(args.local_dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        processed_texts = []
        for item in raw_data:
            if "conversations" in item:
                text_content = " ".join([conv["value"] for conv in item["conversations"] if "value" in conv])
                if text_content: processed_texts.append({"text": text_content})
        raw_datasets = Dataset.from_list(processed_texts).train_test_split(test_size=0.01, seed=42)
        
        def tokenize_function(examples):
            examples["text"] = [str(t) for t in examples["text"]]
            tokenized = tokenizer(examples["text"])
            concatenated_examples = {k: sum(tokenized[k], []) for k in tokenized.keys()}
            total_length = len(concatenated_examples[list(tokenized.keys())[0]])
            total_length = (total_length // args.seq_len) * args.seq_len
            result = {
                k: [t[i : i + args.seq_len] for i in range(0, total_length, args.seq_len)]
                for k, t in concatenated_examples.items()
            }
            return result
        tokenized_datasets = raw_datasets.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

    # 创建 Jittor DataLoader
    train_dataset = JittorTextDataset(tokenized_datasets["train"])
    
    # --- [DDP 修改 3] 使用 DistributedSampler ---
    train_sampler = jt.dataset.DistributedSampler(train_dataset, world_size, rank, shuffle=True)
    dataloader = train_dataset.set_attrs(
        batch_size=args.batch_size, 
        sampler=train_sampler
    )
    print(f"Rank {rank}: Dataset ready. Total samples on this GPU: {len(dataloader)}")


    # --- 5. 设置优化器和训练循环 ---
    if rank == 0: print("\n--- Step 5: Starting Training Loop ---")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    medusa_decay = 0.8 
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch) # 确保每轮 epoch 的 shuffle 不同
        for batch_idx, input_ids in enumerate(dataloader):
            medusa_logits = model.execute(input_ids=input_ids, medusa_forward=True, output_orig=False)
            
            total_loss = jt.zeros(1)
            for k in range(args.medusa_heads):
                head_logits = medusa_logits[k]
                offset = k + 1
                if offset >= input_ids.shape[1]: break
                pred = head_logits[:, :-offset, :]
                target = input_ids[:, offset:]
                loss_k = nn.cross_entropy_loss(pred.reshape(-1, llama_config.vocab_size), target.reshape(-1))
                total_loss += loss_k * (medusa_decay ** k)
            
            # Jittor 的 optimizer.step 会自动处理梯度同步
            optimizer.step(total_loss)
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {batch_idx} | Loss: {total_loss.item():.4f}")

    # --- 6. 保存训练好的 Medusa Heads ---
    # --- [DDP 修改 4] 只有主进程保存模型 ---
    if rank == 0:
        print("\n--- Step 6: Saving Trained Medusa Heads ---")
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "medusa_lm_head.jtr")
        jt.save(model.medusa_head.parameters(), save_path)
        print(f"Medusa heads saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--jittor_weights_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./medusa_checkpoints")
    # 新增本地数据集路径参数
    parser.add_argument("--local_dataset_path", type=str, required=True, help="Path to the local JSON dataset file (e.g., ShareGPT_Vicuna_unfiltered.json)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--medusa_heads", type=int, default=5)

    args = parser.parse_args()
    main(args)