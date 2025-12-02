# python weight_convert.py     --model_path ../Medusa/vicuna-7b-v1.3     --save_path ./vicuna-jittor-weights/vicuna-7b-v1.3.jtr


# --- 步骤 1: 导入库 ---
print("[DEBUG] Script started.")

# 将最可能引起冲突的库放在最前面
import torch
print("[DEBUG] Imported torch.")

import jittor as jt
print("[DEBUG] Imported jittor.")

import os
print("[DEBUG] Imported os.")

import argparse
print("[DEBUG] Imported argparse.")

from safetensors.torch import load_file
print("[DEBUG] Imported from safetensors.torch.")

import json
print("[DEBUG] Imported json.")

print("\n[DEBUG] All imports successful. Defining convert function...")

# --- 步骤 2: 定义函数 ---
def convert(model_path, save_path):
    print(f"\n[DEBUG] Entered convert function. model_path = {model_path}")
    
    print(f"Loading weights from {model_path}...")
    
    final_state_dict = {}
    
    index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
    print(f"[DEBUG] Checking for index file: {index_file}")

    if os.path.exists(index_file):
        print("Detected sharded .bin format (multiple files).")
        
        with open(index_file, 'r') as f:
            print("[DEBUG] Reading index file...")
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        
        sharded_files = {}
        for tensor_name, file_name in weight_map.items():
            if file_name not in sharded_files:
                sharded_files[file_name] = []
            sharded_files[file_name].append(tensor_name)
            
        print("[DEBUG] Grouped tensors by file name.")
        
        for file_name, tensor_names in sharded_files.items():
            file_path = os.path.join(model_path, file_name)
            print(f"\n[DEBUG] About to load torch shard: {file_path}")
            # >>> 崩溃可能发生在这里 <<<
            pt_state_dict_shard = torch.load(file_path, map_location="cpu")
            print(f"  - Successfully loaded {file_name}.")
            
            for tensor_name in tensor_names:
                if tensor_name in pt_state_dict_shard:
                    final_state_dict[tensor_name] = pt_state_dict_shard[tensor_name]
                else:
                    print(f"Warning: Tensor '{tensor_name}' not found in shard '{file_name}'")
        print("[DEBUG] Finished loading all shards.")

    else:
        # ... (单文件逻辑) ...
        print("[DEBUG] Sharded model not found. Looking for single file model.")
        raise NotImplementedError("Single file logic not shown for brevity, assuming sharded model for now.")

    print("\n[DEBUG] Converting all tensors to Jittor format...")
    jittor_state_dict = {}
    for key, tensor in final_state_dict.items():
        np_arr = tensor.cpu().float().numpy()
        jittor_state_dict[key] = np_arr
    print("[DEBUG] Conversion to NumPy arrays successful.")

    # ... (Medusa head logic) ...

    print(f"\n[DEBUG] About to save Jittor weights to {save_path}...")
    jt.save(jittor_state_dict, save_path)
    print("Done!")


# --- 步骤 3: 主程序入口 ---
if __name__ == "__main__":
    print("\n[DEBUG] Inside __main__ block.")
    
    parser = argparse.ArgumentParser()
    print("[DEBUG] Parser created.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model folder")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save jittor weights")
    print("[DEBUG] Arguments defined.")
    
    args = parser.parse_args()
    print(f"[DEBUG] Arguments parsed: {args}")
    
    # --- 步骤 4: 调用主函数 ---
    print("\n[DEBUG] About to call the convert function...")
    convert(args.model_path, args.save_path)