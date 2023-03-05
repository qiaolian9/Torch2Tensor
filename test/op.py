
import tvm
from tvm.relax.expr import Expr
from typing import List, Optional, Tuple, Union
from tvm import relax
from tvm.script import relax as R
bb = relax.BlockBuilder()

@tvm.ir.op.register_op_attr('aadd', 'FInferStructInfo')
def add(call: relax.Call, bb):
    x = call.args[0]
    y = call.args[0]
    z = x + y
    type(z)
    return z.struct_info
@tvm.register_func('addd')
def add_(x, y):
    return x + y

with bb.function('main'):
    with bb.dataflow():
        x = relax.Var('x', R.Tensor((1,3)))
        y = relax.Var('y', R.Tensor((1,3)))

        d = tvm.ir.Op.get("aadd")
        # d = tvm.ir.Op.get("relax.add")
        print(d.get_attr('FInferStructInfo'))
        c = relax.Call(d, [x, y])
        out = bb.emit(c)
        bb.emit_output(out)
    
    bb.emit_func_output(out, [x ,y])

print(bb.get().show())



