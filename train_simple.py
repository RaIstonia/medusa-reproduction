import jittor as jt
from jittor import nn, optim
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig
import numpy as np

# 模拟数据集
class DummyDataset(jt.dataset.Dataset):
    def __init__(self, vocab_size, seq_len, length=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        
    def __getitem__(self, i):
        # 随机生成数据
        x = np.random.randint(0, self.vocab_size, (self.seq_len,)).astype(np.int64)
        return x # input_ids
        
    def __len__(self):
        return self.length

def train_step():
    jt.flags.use_cuda = 1
    
    # 1. 配置与模型
    vocab_size = 1000
    hidden_size = 64
    heads = 4
    medusa_heads = 3 # 预测未来 3 个 token
    
    llama_config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, 
        num_attention_heads=heads, num_key_value_heads=heads,
        num_hidden_layers=2, intermediate_size=128
    )
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_heads, medusa_num_layers=1,
        hidden_size=hidden_size, vocab_size=vocab_size
    )
    
    print("Building Model...")
    base_model = LlamaForCausalLM(llama_config)
    # [关键] 冻结 Base Model 参数 (Stage 1 Training)
    for p in base_model.parameters():
        p.stop_grad()
        
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train() # 切换到训练模式
    
    # 2. 优化器 (只优化 Medusa Heads)
    # 过滤出 requires_grad 的参数 (Jittor 中默认都求导，除非 stop_grad)
    # 但我们上面已经对 base_model 做了 stop_grad
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. 数据集
    dataloader = DummyDataset(vocab_size, seq_len=20).set_attrs(batch_size=4, shuffle=True)
    
    # 4. Loss 参数
    medusa_decay = 0.8 # 越远的头权重越小
    
    print("Start Training Loop...")
    for epoch in range(2):
        for batch_idx, input_ids in enumerate(dataloader):
            # input_ids: [Batch, Seq]
            
            # Forward Pass
            # medusa_forward=True, 且不需要 output_orig (训练时只算 Head Loss)
            # 输出形状: [Medusa_Heads, Batch, Seq, Vocab]
            medusa_logits = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=False 
            )
            
            # 计算 Loss
            # Target 构造:
            # Head k (k=0...N-1) 预测的是 input_ids[t + k + 1]
            
            total_loss = jt.zeros(1)
            
            for k in range(medusa_heads):
                # 获取第 k 个头的 logits: [Batch, Seq, Vocab]
                head_logits = medusa_logits[k]
                
                # 对应的 Target
                # 预测位置: input_ids 的第 k+1 个位置开始
                # Logits 有效部分: 0 到 Seq-(k+1)
                # Labels 有效部分: k+1 到 Seq
                
                # 举例: Seq=10, k=0 (预测下一词)
                # Logits: predict at t=0...8
                # Labels: real token at t=1...9
                
                offset = k + 1
                if offset >= input_ids.shape[1]:
                    break
                    
                # 切片
                pred = head_logits[:, :-offset, :] # [Batch, Seq-Offset, Vocab]
                target = input_ids[:, offset:]     # [Batch, Seq-Offset]
                
                # Flatten
                pred = pred.reshape(-1, vocab_size)
                target = target.reshape(-1)
                
                # Cross Entropy
                loss_k = nn.cross_entropy_loss(pred, target)
                
                # 加权累加
                weight = medusa_decay ** k
                total_loss += loss_k * weight
            
            # Backward
            optimizer.step(total_loss)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Step {batch_idx} Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    train_step()