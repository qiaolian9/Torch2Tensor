import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax

@tvm.ir.op.register_op_attr('relax.nn.hardswish', 'FInferStructInfo')
def hardswish_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    return relax.TensorStructInfo(data_.struct_info.shape)

@tvm.register_func('relax.op.nn.hardswish')
def hardswish_func(data):
    op =  tvm.ir.Op.get('relax.nn.hardswish')     
    return relax.Call(op, [data])

def hardswish(
    data: Expr,
) -> Expr:
    return hardswish_func(data)
