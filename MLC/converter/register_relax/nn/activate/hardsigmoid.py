import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax

@tvm.ir.op.register_op_attr('relax.hardsigmoid', 'FInferStructInfo')
def hardsigmoid_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    return relax.TensorStructInfo(data_.struct_info.shape)

@tvm.register_func('relax.op.hardsigmoid')
def hardsigmoid_func(data):
    op =  tvm.ir.Op.get('relax.hardsigmoid')     
    return relax.Call(op, [data])

def hardsigmoid(
    data: Expr,
) -> Expr:
    return hardsigmoid_func(data)
