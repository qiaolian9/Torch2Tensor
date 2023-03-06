import tvm
from tvm.relax.expr import Expr
from tvm import relax

@tvm.ir.op.register_op_attr('relax.contiguous', 'FInferStructInfo')
def contiguous_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    return relax.TensorStructInfo(data_.struct_info.shape)

@tvm.register_func('relax.op.contiguous')
def contiguous_func(data):
    
    op =  tvm.ir.Op.get('relax.contiguous')       
    return relax.Call(op, [data])

def contiguous(
    data: Expr,
) -> Expr:

    return contiguous_func(data)
