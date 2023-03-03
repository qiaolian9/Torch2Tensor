# import tvm
# from tvm.relax.expr import Expr
# from typing import List, Optional, Tuple, Union
# from tvm import relax

# @tvm.ir.op.register_op_attr('relax.nn.avg_pool2d', 'FInferStructInfo')
# def avgpool2d(data, pool_size):
#     print('8'* 10, data, type(data))
#     return data

# @tvm.register_func('relax.op.nn.avg_pool2d')
# def avg_pool2d_func(data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout):
#     avg_pool2d_attrs = dict(
#     pool_size = pool_size,
#     strides = strides,
#     padding = padding,
#     dilation = dilation,
#     ceil_mode = ceil_mode,
#     layout = layout,
#     out_layout = out_layout
#     )
#     print('7' * 8, data, type(data))
#     op =  tvm.ir.Op.get('relax.nn.avg_pool2d')
#     print(op.get_attr('FInferStructInfo'))
#     avg_pool2d_attrs = tvm.ir.make_node('DictAttrs', **avg_pool2d_attrs)        
#     return relax.Call(op, [data], avg_pool2d_attrs)

# def avg_pool2d(
#     data: Expr,
#     pool_size: Union[int, Tuple[int, int]] = (1, 1),
#     strides: Union[int, Tuple[int, int]] = (1, 1),
#     padding: Union[int, Tuple[int, ...]] = (0, 0),
#     dilation: Union[int, Tuple[int, int]] = (1, 1),
#     ceil_mode: bool = False,
#     layout: str = "NCHW",
#     out_layout: Optional[str] = None
# ) -> Expr:

#     if isinstance(pool_size, int):
#             pool_size = (pool_size, pool_size)
#     if isinstance(strides, int):
#         strides = (strides, strides)
#     if isinstance(dilation, int):
#         dilation = (dilation, dilation)
#     if isinstance(padding, int):
#         padding = (padding, padding, padding, padding)

#     return avg_pool2d_func(data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout)
