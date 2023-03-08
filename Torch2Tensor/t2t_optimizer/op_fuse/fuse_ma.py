from tvm import relax, IRModule
from tvm.relax import Function
from typing import Optional
import tvm
relax.transform.EwiseFuseFMA

relax.op.linear

@relax.expr_functor.mutator
class DenseAddFusor(relax.PyExprMutator):
    def __init__(self, mod: Optional[IRModule] = None) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.matmul_op = tvm.ir.Op.get('relax.matmul')
        self.add_op = tvm.ir.Op.get('relax.add')
        self.count = 0

    def create_denseadd_op(self, call: relax.Call, value: relax.Call, fn_name=None):
        b = call.args[1]
        x = value.args[0]
        w = value.args[1]

        x_ = relax.Var('x', x.struct_info)
        w_ = relax.Var('w', w.struct_info)
        b_ = relax.Var('b', b.struct_info)

        bb = relax.BlockBuilder()
        with bb.function(fn_name, [x_, w_, b_]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.matmul(x_, w_), name_hint='fuse_matmul')
                lv1 = bb.emit(relax.op.add(lv0, b_), name_hint='fuse_matmul')
                gv = bb.emit_output(lv1)
            bb.emit_func_output(gv)
        
        fused_fn = bb.get()[fn_name].with_attr('Primitive', 1)
        return fused_fn, x, w, b


    def transform(self):
        # optimizer every relax function
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                self.builder_.add_func(func, func_name=global_var.name_hint)
            if func.attrs is not None and 'Primitive' in func.attrs and func['Primitive'] != 0:
                self.builder_.add_func(func, func_name=global_var.name_hint)

            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()
    
    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op

        if not match_call(call, self.add_op):
            return call
        
        value = self.lookup_binding(call.args[0])
        if value is None or not match_call(value, self.matmul_op):
            return call
        
        fn_name = 'fused_dense_add%d' % self.count
        self.count += 1
        fused_fn, x, w, b = self.create_denseadd_op(call, value, fn_name)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        return relax.Call(global_var, [x, w, b])

@tvm.ir.transform.module_pass(opt_level=2, name="DeseAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return DenseAddFusor(mod).transform()
