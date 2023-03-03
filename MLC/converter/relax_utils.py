from __future__ import annotations
from tvm import relax
import torch.nn as nn

# torch graph/nn.Module --> IR high level relax.function
def map_params(param: nn.Parameter):
    return relax.const(param.data.cpu().numpy(), dtype='float32')

def fetch_attr(fx_mod, target: str):
    # 获取mod属性
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr