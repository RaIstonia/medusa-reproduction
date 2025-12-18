import jittor as jt
from jittor import nn, optim
# 注意这里根据你的实际文件名修改 import
from medusa.model.modeling_medusa import MedusaConfig, MedusaModel
from medusa.model.modeling_llama import LlamaForCausalLM, LlamaConfig
import numpy as np
import time

# [修改 1] 过拟合数据集：每次返回固定的序列
class OverfitDataset(jt.dataset.Dataset):
    def __init__(self, vocab_size, seq_len, length=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        # 固定一个 pattern，例如 [0, 1, 2, 3, ...]
        self.fixed_data = np.arange(seq_len) % vocab_size
        self.fixed_data = self.fixed_data.astype(np.int64)
        
    def __getitem__(self, i):
        # 无论 index 是多少，都返回同样的数据
        return self.fixed_data
        
    def __len__(self):
        return self.length

def train_step():
    jt.flags.use_cuda = 1
    
    # 配置
    vocab_size = 1000
    hidden_size = 64
    heads = 4
    medusa_heads = 3
    
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
    # 冻结 Base
    for p in base_model.parameters():
        p.stop_grad()
        
    model = MedusaModel(medusa_config, base_model=base_model)
    model.train()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-2) # [修改 2] 加大一点 LR 方便观察下降
    
    # [修改 3] 使用 OverfitDataset
    dataloader = OverfitDataset(vocab_size, seq_len=20, length=200).set_attrs(batch_size=16, shuffle=True)
    
    medusa_decay = 0.8
    
    print("Start Overfitting Test...")
    start_time = time.time()
    
    for epoch in range(5): # 跑 5 个 epoch
        total_epoch_loss = 0
        steps = 0
        for batch_idx, input_ids in enumerate(dataloader):
            # input_ids: [Batch, Seq]
            
            medusa_logits = model.execute(
                input_ids=input_ids,
                medusa_forward=True,
                output_orig=False 
            )
            
            total_loss = jt.zeros(1)
            
            for k in range(medusa_heads):
                head_logits = medusa_logits[k]
                offset = k + 1
                if offset >= input_ids.shape[1]: break
                    
                pred = head_logits[:, :-offset, :]
                target = input_ids[:, offset:]
                
                pred = pred.reshape(-1, vocab_size)
                target = target.reshape(-1)
                
                loss_k = nn.cross_entropy_loss(pred, target)
                weight = medusa_decay ** k
                total_loss += loss_k * weight
            
            optimizer.step(total_loss)
            
            total_epoch_loss += total_loss.item()
            steps += 1
            
        avg_loss = total_epoch_loss / steps
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # 验证：Loss 应该大幅下降
    if avg_loss < 1.0:
        print("SUCCESS: Model successfully overfitted (Loss < 1.0). Training logic is correct!")
    else:
        print("WARNING: Loss did not drop significantly. Check optimizer or gradient.")

if __name__ == "__main__":
    train_step()