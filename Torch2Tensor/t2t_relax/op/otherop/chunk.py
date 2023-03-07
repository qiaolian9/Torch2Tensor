import tvm
from tvm.relax.expr import Expr
from tvm import relax

@tvm.ir.op.register_op_attr('relax.chunk', 'FInferStructInfo')
def chunk_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    attrs = call.attrs
    chunks = attrs.chunks
    dim = attrs.dim
    out_shape = []
    axis = data_.struct_info.shape[dim.value].value

    if axis % chunks == 0:
        num = axis // chunks
        shape = []
        for i in data_.struct_info.shape:
            shape.append(i.value)
        shape[dim.value] = num
        out_shape = [relax.TensorStructInfo(shape)] * chunks.value
    else:
        raise Warning('to be developed')
    return relax.TupleStructInfo(out_shape)

@tvm.register_func('relax.op.chunk')
def chunk_func(data, chunks, dim):
    chunk_attrs = dict(
        chunks=chunks,
        dim=dim
        )
    op =  tvm.ir.Op.get('relax.chunk')
    chunk_attrs = tvm.ir.make_node('DictAttrs', **chunk_attrs)        
    return relax.Call(op, [data], chunk_attrs)

def chunk(
    data: Expr,
    chunks,
    dim,
) -> Expr:

    return chunk_func(data, chunks, dim)
