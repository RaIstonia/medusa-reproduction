import jittor as jt
from jittor import nn, optim
import os
import argparse
import json
from transformers import AutoTokenizer
from datasets import load_dataset

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
    jt.flags.use_cuda = 1
    
    # --- 1. 加载预训练的基座模型 (最关键的一步) ---
    print("--- Step 1: Loading Pre-trained Base Model ---")
    
    # 从 Hugging Face 模型目录加载 Llama 配置
    with open(os.path.join(args.base_model_path, "config.json"), 'r') as f:
        llama_config_dict = json.load(f)
    llama_config = LlamaConfig.from_dict(llama_config_dict)

    # 初始化一个空的 Llama 模型结构
    base_model = LlamaForCausalLM(llama_config)
    
    # 加载您之前转换好的 Jittor 权重
    print(f"Loading converted Jittor weights from: {args.jittor_weights_path}")
    base_model.load_parameters(jt.load(args.jittor_weights_path))
    print("Base model weights loaded successfully.")

    # --- 2. 冻结基座模型的所有参数 ---
    print("\n--- Step 2: Freezing Base Model Parameters ---")
    for param in base_model.parameters():
        param.stop_grad() # 阻止Jittor计算这些参数的梯度

    # --- 3. 组装完整的 Medusa 模型 ---
    print("\n--- Step 3: Assembling Medusa Model ---")
    # 创建一个与基座模型兼容的 Medusa 配置
    medusa_config = MedusaConfig(
        medusa_num_heads=args.medusa_heads,
        medusa_num_layers=1,
        hidden_size=llama_config.hidden_size, # 必须匹配
        vocab_size=llama_config.vocab_size     # 必须匹配
    )
    
    # 将预训练且已冻结的 base_model 传入 MedusaModel
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train() # 设置为训练模式
    print("Medusa model assembled.")
    
    # --- 4. 准备真实文本数据集 ---
    print("\n--- Step 4: Preparing Dataset ---")
    # 加载与基座模型匹配的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # 加载一个标准的数据集，例如 'wikitext'
    # 'wikitext-2-raw-v1' 比较小，适合快速测试
    print("Loading 'wikitext-2-raw-v1' dataset...")
    raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # 定义一个函数来对数据集进行分词和分块
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"])
        # 将所有文本拼接起来，然后切分成固定长度的块
        concatenated_examples = {k: sum(tokenized[k], []) for k in tokenized.keys()}
        total_length = len(concatenated_examples[list(tokenized.keys())[0]])
        total_length = (total_length // args.seq_len) * args.seq_len
        
        result = {
            k: [t[i : i + args.seq_len] for i in range(0, total_length, args.seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing and chunking dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    
    # 创建 Jittor DataLoader
    train_dataset = JittorTextDataset(tokenized_datasets["train"])
    dataloader = train_dataset.set_attrs(batch_size=args.batch_size, shuffle=True)
    print(f"Dataset ready. Total training samples: {len(train_dataset)}")

    # --- 5. 设置优化器和训练循环 ---
    print("\n--- Step 5: Starting Training Loop ---")
    # 优化器只会更新那些没有被 stop_grad() 的参数，也就是 Medusa Heads 的参数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss 参数
    medusa_decay = 0.8 
    
    for epoch in range(args.epochs):
        for batch_idx, input_ids in enumerate(dataloader):
            # 前向传播
            medusa_logits = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=False 
            )
            
            # 计算 Loss (这部分逻辑与您原来的一致，是正确的)
            total_loss = jt.zeros(1)
            for k in range(args.medusa_heads):
                head_logits = medusa_logits[k]
                offset = k + 1
                if offset >= input_ids.shape[1]:
                    break
                pred = head_logits[:, :-offset, :]
                target = input_ids[:, offset:]
                
                loss_k = nn.cross_entropy_loss(pred.reshape(-1, llama_config.vocab_size), target.reshape(-1))
                total_loss += loss_k * (medusa_decay ** k)
            
            # 反向传播和参数更新
            optimizer.step(total_loss)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {batch_idx} | Loss: {total_loss.item():.4f}")

    # --- 6. 保存训练好的 Medusa Heads ---
    print("\n--- Step 6: Saving Trained Medusa Heads ---")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    save_path = os.path.join(args.output_dir, "medusa_lm_head.jtr")
    # 只保存 medusa_head 的参数
    jt.save(model.medusa_head.parameters(), save_path)
    print(f"Medusa heads saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the original Hugging Face model folder (e.g., ../Medusa/vicuna-7b-v1.3)")
    parser.add_argument("--jittor_weights_path", type=str, required=True, help="Path to the converted Jittor base model weights file (*.jtr)")
    parser.add_argument("--output_dir", type=str, default="./medusa_checkpoints", help="Directory to save the trained Medusa heads")
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for training")
    parser.add_argument("--medusa_heads", type=int, default=5, help="Number of Medusa heads to train")

    args = parser.parse_args()
    main(args)