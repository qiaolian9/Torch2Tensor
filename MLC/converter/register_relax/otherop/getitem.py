import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax

@tvm.ir.op.register_op_attr('relax.get_item', 'FInferStructInfo')
def get_item_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    index = call.attrs.index
    
    return relax.TensorStructInfo([1], dtype='int32')

@tvm.register_func('relax.op.get_item')
def get_item_func(data, index):
    op =  tvm.ir.Op.get('relax.get_item') 
    index_attrs = dict(index = index)
    index_attrs = tvm.ir.make_node('DictAttrs', **index_attrs)    
    return relax.Call(op, [data], index_attrs)

def get_item(
    data: Expr,
    index: relax.expr.PrimExpr
) -> Expr:
    return get_item_func(data, index)
