import tvm
from tvm import relax, IRModule
from typing import List, Optional, Tuple, Union
from .op_map import *
from loguru import logger

@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: Optional[IRModule], op_map=None):
        # print(op_map)
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }
    
    def visit_call_(self, call: relax.Call):
        call = self.visit_expr_post_order(call)
            
        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call
    
    def transform(self):
        for global_val, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_fn = self.visit_expr(func)
            updated_fn = relax.analysis.remove_all_unused(updated_fn)
            self.builder_.update_func(global_val, updated_fn)

        return self.builder_.get()

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    def __init__(self,):
        from .op_map import DEFAULT_MAP_PATTERNS
        self.op_map = DEFAULT_MAP_PATTERNS
        self.fuseTIR = relax.transform.FuseTIR()
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        mod = LowerToTensorIR(mod, op_map=DEFAULT_MAP_PATTERNS).transform()
        return self.fuseTIR(mod)
    
