from __future__ import annotations
import re
import torch
import torch.nn as nn
import tvm
from tvm import relax


def get_shape(obj):
    return obj["shape"]


def get_dtype(obj):
    return obj["dtype"]


def map_reduce(args, fn, device=None):
    shape_list = []
    if isinstance(args, tuple):
        shape = sum(list(map_reduce(elem, fn, device) for elem in args), [])
    elif isinstance(args, list):
        shape = sum(list(map_reduce(elem, fn, device) for elem in args), [])
    elif args is not None:
        if device is not None:
            shape = [fn(args, device)]
        else:
            shape = [fn(args)]
    else:
        shape = []

    shape_list.extend(shape)

    return shape_list

def get_function_name(node_target):
    function_name = re.findall(
        r"(?:function|method) ([a-z|_|0-9]+.*?)", str(node_target)
    )[0]

    return function_name

def map_replace(args, fn):
    if all(isinstance(element, int) for element in args):
        return fn(args)
    elif isinstance(args, tuple):
        return list(map_replace(elem, fn) for elem in args)
    elif isinstance(args, list):
        return list(map_replace(elem, fn) for elem in args)
    else:
        return fn(args)

def get_torch_size(obj):
    return torch.Size(obj)

def gen_torch_tensor(obj):
    return torch.rand(obj).to(torch.int32)

def gen_numpy_data(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj.numpy()

def gen_tvm_data(obj, device=tvm.cpu()):
    obj = obj.numpy()
    return tvm.nd.array(obj, device)

def gen_torch_data(obj, device=torch.device('cuda:0')):
    return obj.to(device)

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
