import torch
import os
import safetensors
from typing import Dict, List, Optional, Set, Tuple
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file
 
def ckpt2safetensors():
    loaded = torch.load('v1-5-pruned-emaonly.ckpt')
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    safetensors.torch.save_file(loaded, 'v1-5-pruned-emaonly.safetensors')
 
def st2ckpt():
    # 加载 .safetensors 文件
    data = safetensors.torch.load_file('v1-5-pruned-emaonly.safetensors.bk')
    data["state_dict"] = data
    # 将数据保存为 .ckpt 文件
    torch.save(data, os.path.splitext('v1-5-pruned-emaonly.safetensors')[0] + '.ckpt')