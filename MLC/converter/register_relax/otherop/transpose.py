import tvm
from tvm.relax.expr import Expr
from tvm.ir import Attrs
from tvm import relax
from tvm.runtime import Object

@tvm.ir.op.register_op_attr('relax.transpose', 'FInferStructInfo')
def transpose_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    attrs = call.attrs
    dim1 = attrs.dim1
    dim2 = attrs.dim2
    data_shape = data_.struct_info.shape
    out_shape = []
    for i in data_shape:
        out_shape.append(i.value)
    out_shape[dim1.value] = data_shape[dim2.value].value
    out_shape[dim2.value] = data_shape[dim1.value].value
    
    return relax.TensorStructInfo(out_shape)

@tvm.register_func('relax.op.transpose')
def transpose_func(data, dim1, dim2):
    axes = [i for i in range(len(data.struct_info.shape))]
    axes[dim1] = dim2
    axes[dim2] = dim1
    transpose_attrs = dict(
        dim1 = dim1,
        dim2 = dim2,
        axes = tuple(axes)
        )
    op =  tvm.ir.Op.get('relax.transpose')
    transpose_attrs = tvm.ir.make_node('DictAttrs', **transpose_attrs)        
    return relax.Call(op, [data], transpose_attrs)

def transpose(
    data: Expr,
    dim1,
    dim2
) -> Expr:

    return transpose_func(data, dim1, dim2)
