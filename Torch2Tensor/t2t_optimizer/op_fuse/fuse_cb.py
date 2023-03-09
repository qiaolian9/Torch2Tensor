from tvm import relax, IRModule
from tvm.relax import Function
from typing import Optional
import tvm
relax.transform.EwiseFuseFMA

__all__ = ['FuseCBPass']

@relax.expr_functor.mutator
class ConvBFusor(relax.PyExprMutator):
    def __init__(self, mod: Optional[IRModule] = None) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.conv_op = tvm.ir.Op.get('relax.nn.conv2d')
        self.bn_op = tvm.ir.Op.get('relax.nn.batch_norm')
        self.count = 0

    def create_cb_op(self, call: relax.Call, value: relax.Call, fn_name=None):
        x = value.args[0]
        # bn
        gamma = call.args[1]
        beta = call.args[2]
        moving_mean = call.args[3]
        moving_var = call.args[4]
        bn_attrs = call.attrs
        # conv
        kernel = value.args[1]
        conv_attrs = value.attrs

        x_ = relax.Var('x', x.struct_info)
        gamma_ = relax.Var('gamma', gamma.struct_info)
        beta_ = relax.Var('beta', beta.struct_info)
        moving_mean_ = relax.Var('moving_mean', moving_mean.struct_info)
        moving_var_ = relax.Var('moving_var', moving_var.struct_info)
        kernel_ = relax.Var('kernel', kernel.struct_info)
        
        cb_attrs = dict(
            axis = bn_attrs.axis,
            epsilon = bn_attrs.epsilon,
            center = bn_attrs.center,
            scale = bn_attrs.scale,
            strides = conv_attrs.strides,
            padding = conv_attrs.padding,
            dilation = conv_attrs.dilation,
            groups = conv_attrs.groups,
            data_layout = conv_attrs.data_layout,
            kernel_layout = conv_attrs.kernel_layout,
            out_layout = conv_attrs.out_layout,
            out_dtype = conv_attrs.out_dtype,
        )
        cb_attrs = tvm.ir.make_node('DictAttrs', **cb_attrs)

        bb = relax.BlockBuilder()
        with bb.function(fn_name, [x_, gamma_, beta_, moving_mean_, moving_var_, kernel_]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.nn.conv2d(x_, kernel_, **conv_attrs))
                lv1 = bb.emit(relax.op.nn.batch_norm(lv0, gamma_, beta_, moving_mean_, moving_var_, **bn_attrs))
                gv = bb.emit_output(lv1)
            bb.emit_func_output(gv)

        fused_fn = bb.get()[fn_name].with_attr('Primitive', 1)
        return fused_fn, x, gamma, beta, moving_mean, moving_var, kernel, cb_attrs

    def transform(self):
        # optimizer every relax function
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is not None and 'Primitive' in func.attrs and func.attrs['Primitive'] != 0:
                continue

            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()
    
    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            elif isinstance(op, list):
                return node.op in op
            return node.op == op

            
        if not match_call(call, self.bn_op):
            return call
        
        value = self.lookup_binding(call.args[0])
        if value is None or not match_call(value, self.conv_op):
            return call
        
        fn_name = 'fused_cb%d' % self.count
        self.count += 1
        fused_fn, x, gamma, beta, moving_mean, moving_var, kernel, cb_attrs \
                    = self.create_cb_op(call, value, fn_name)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        return relax.Call(global_var, [x, gamma, beta, moving_mean, moving_var, kernel], cb_attrs)

@tvm.ir.transform.module_pass(opt_level=2, name="CBFuse")
class FuseCBPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return ConvBFusor(mod).transform()
