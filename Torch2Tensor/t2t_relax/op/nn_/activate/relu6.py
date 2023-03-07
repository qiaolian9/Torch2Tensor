import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax

@tvm.ir.op.register_op_attr('relax.nn.relu6', 'FInferStructInfo')
def relu6_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    return relax.TensorStructInfo(data_.struct_info.shape)

@tvm.register_func('relax.op.nn.relu6')
def relu6_func(data):
    op =  tvm.ir.Op.get('relax.nn.relu6')     
    return relax.Call(op, [data])

def relu6(
    data: Expr,
) -> Expr:
    return relu6_func(data)
