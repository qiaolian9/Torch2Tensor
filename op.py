
import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax
from tvm.script import relax as R

# op_ = tvm._ffi.registry.list_global_func_names()
# op_nn = []
# for i in op_:
#     if 'relax.op.nn' in i:
#         print(i)

# print(tvm.get_global_func('relax.op.nn.avg_pool2d'))
tvm.ir.op.register_op_attr('relax.nn.avg_pool2d', 'avg_pool2d', '1')
a = tvm.ir.Op.get('relax.nn.avg_pool2d')
print(a, type(a))

b = dict(a=1,b=2)
b = tvm.ir.make_node('DictAttrs', **b)
print(b, type(b))

datta = relax.Var('data', R.Tensor([1]))
c = relax.Call(a, [datta], b)
print(c, type(c), c.op, c.attrs.a)

import torch 

torch.nn.av
