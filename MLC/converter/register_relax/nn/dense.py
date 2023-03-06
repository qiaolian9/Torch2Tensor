import tvm
from tvm.relax.expr import Expr
from typing import Optional, Union
from tvm import relax
from tvm import DataType

@tvm.ir.op.register_op_attr('relax.nn.dense', 'FInferStructInfo')
def dense_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    weight_ = call.args[1]

    kernel_shape = weight_.struct_info.shape
    data_shape = data_.struct_info.shape
    out_shape = []
    for i in data_shape:
        out_shape.append(i)
    out_shape[-1] = kernel_shape[0]

    return relax.TensorStructInfo(out_shape)

@tvm.register_func('relax.op.nn.dense')
def dense_func(data, weight, bias, out_dtype):
    op =  tvm.ir.Op.get('relax.nn.dense') 
    matmul_attrs = tvm.ir.make_node('relax.attrs.MatmulAttrs', **dict(out_dtype=out_dtype))
    return relax.Call(op, [data, weight, bias], matmul_attrs)

def dense(
    data: Expr,
    weight: Expr,
    bias: Optional[Expr] = None,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:


    return dense_func(data, weight, bias, out_dtype)
