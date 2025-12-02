import torch
import jittor as jt
import os
import argparse
from safetensors.torch import load_file
import json # 需要导入 json 库

def convert(model_path, save_path):
    print(f"Loading weights from {model_path}...")
    
    final_state_dict = {} # 这是我们最终要保存的 Jittor state dict
    
    # 1. 检查是否存在分片模型的索引文件
    index_file = os.path.join(model_path, "pytorch_model.bin.index.json")

    if os.path.exists(index_file):
        print("Detected sharded .bin format (multiple files).")
        
        # 读取索引文件
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # index_data["weight_map"] 是一个 {"tensor_name": "filename.bin"} 的字典
        weight_map = index_data["weight_map"]
        
        # 为了高效，我们将按文件名分组，避免重复加载同一个文件
        sharded_files = {}
        for tensor_name, file_name in weight_map.items():
            if file_name not in sharded_files:
                sharded_files[file_name] = []
            sharded_files[file_name].append(tensor_name)
            
        # 逐个加载分片文件并提取所需的张量
        for file_name, tensor_names in sharded_files.items():
            file_path = os.path.join(model_path, file_name)
            print(f"  - Loading {file_name}...")
            # 加载单个分片文件
            pt_state_dict_shard = torch.load(file_path, map_location="cpu")
            
            # 从这个分片中提取我们需要的张量
            for tensor_name in tensor_names:
                # 注意：key 在 weight_map 和 state_dict 中可能不完全一致
                # 我们需要在加载的 state_dict 中找到正确的 key
                # 通常它们是一致的，但为了稳健，可以做个检查
                if tensor_name in pt_state_dict_shard:
                    final_state_dict[tensor_name] = pt_state_dict_shard[tensor_name]
                else:
                    print(f"Warning: Tensor '{tensor_name}' not found in shard '{file_name}'")

    else:
        # 如果不是分片模型，则使用原始逻辑（单文件 .bin 或 .safetensors）
        safetensors_file = os.path.join(model_path, "model.safetensors")
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_file):
            print("Detected single .safetensors format")
            final_state_dict = load_file(safetensors_file)
        elif os.path.exists(bin_file):
            print("Detected single .bin format")
            final_state_dict = torch.load(bin_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"Could not find model weights in {model_path}")

    # 2. 转换逻辑 (现在 final_state_dict 包含了所有张量)
    print("Converting all tensors to Jittor format...")
    jittor_state_dict = {}
    for key, tensor in final_state_dict.items():
        np_arr = tensor.cpu().float().numpy()
        jittor_state_dict[key] = np_arr

    # 3. 加载 Medusa Head (如果独立存在) - 这部分逻辑不变
    medusa_head_file = os.path.join(model_path, "medusa_lm_head.pt")
    if os.path.exists(medusa_head_file):
        print("Found independent medusa_lm_head.pt, merging...")
        medusa_sd = torch.load(medusa_head_file, map_location="cpu")
        for key, tensor in medusa_sd.items():
            jittor_state_dict[key] = tensor.cpu().float().numpy()

    # 4. 保存
    print(f"Saving {len(jittor_state_dict)} tensors to {save_path}...")
    jt.save(jittor_state_dict, save_path)
    print("Done!")