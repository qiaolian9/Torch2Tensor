import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax

@tvm.ir.op.register_op_attr('relax.get_shape_value', 'FInferStructInfo')
def getshapevalue_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    index = call.attrs.rate
    return relax.TensorStructInfo([1])

@tvm.register_func('relax.op.get_shape_value')
def getshapevalue_func(data, index):
    op =  tvm.ir.Op.get('relax.get_shape_value') 
    index_attrs = tvm.ir.make_node('relax.attrs.DropoutAttrs', **dict(rate=index))    
    return relax.Call(op, [data], index_attrs)

def getshapevalue(
    data: Expr,
    index: relax.expr.PrimExpr
) -> Expr:
    return getshapevalue_func(data, index)
