import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax

@tvm.ir.op.register_op_attr('relax.nn.avg_pool2d', 'FInferStructInfo')
def avgpool2d_op(call: relax.Call, bb: relax.BlockBuilder):
    data_ = call.args[0]
    attrs = call.attrs
    if attrs.layout == 'NCHW':
        data_shape = data_.struct_info.shape
    else:
        raise Warning("layout %s is not supported" % attrs.layout)
    input_h = data_shape[2].value
    input_w = data_shape[3].value
    kernel_h = attrs.pool_size[0]
    kernel_w = attrs.pool_size[1]
    padding_h = attrs.padding[0] + attrs.padding[2]
    padding_w = attrs.padding[1] + attrs.padding[3]
    numerator_h = input_h + padding_h - attrs.dilation[0] * (kernel_h - 1) - 1;
    numerator_w = input_w + padding_w - attrs.dilation[1] * (kernel_w - 1) - 1;
    if attrs.ceil_mode:
        numerator_h += attrs.strides[0] - 1;
        numerator_w += attrs.strides[1] - 1;
    output_h = numerator_h // attrs.strides[0] + 1
    output_w = numerator_w // attrs.strides[1] + 1
    out_NCHW_shape = [data_shape[0], data_shape[1], output_h, output_w]
    return relax.TensorStructInfo(out_NCHW_shape)

@tvm.register_func('relax.op.nn.avg_pool2d')
def avg_pool2d_func(data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout):
    avg_pool2d_attrs = dict(
        pool_size = pool_size,
        strides = strides,
        padding = padding,
        dilation = dilation,
        ceil_mode = ceil_mode,
        layout = layout,
        out_layout = out_layout
        )
    op =  tvm.ir.Op.get('relax.nn.avg_pool2d')
    avg_pool2d_attrs = tvm.ir.make_node('relax.attrs.MaxPool2DAttrs', **avg_pool2d_attrs)        
    return relax.Call(op, [data], avg_pool2d_attrs)

def avg_pool2d(
    data: Expr,
    pool_size: Union[int, Tuple[int, int]] = (1, 1),
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    ceil_mode: bool = False,
    layout: str = "NCHW",
    out_layout: Optional[str] = None
) -> Expr:

    if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    return avg_pool2d_func(data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout)
