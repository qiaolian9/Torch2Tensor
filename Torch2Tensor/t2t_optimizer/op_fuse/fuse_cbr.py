from tvm import relax, IRModule
from tvm.relax import Function
from typing import Optional
import tvm

__all__ = ['FuseCBRPass']

@relax.expr_functor.mutator
class ConvBRFusor(relax.PyExprMutator):
    def __init__(self, mod: Optional[IRModule] = None) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.conv_op = tvm.ir.Op.get('relax.nn.conv2d')
        self.bn_op = tvm.ir.Op.get('relax.nn.batch_norm')
        act = ['relax.nn.relu', 'relax.nn.relu6', 'relax.nn.silu', 'relax.nn.hardswish']
        self.act = [tvm.ir.Op.get(i) for i in act]
        self.count = 0

    def create_cbr_op(self, call: relax.Call, value_1: relax.Call, value_2: relax.Call, fn_name=None):
        x = value_2.args[0]
        # bn
        gamma = value_1.args[1]
        beta = value_1.args[2]
        moving_mean = value_1.args[3]
        moving_var = value_1.args[4]
        bn_attrs = value_1.attrs
        # conv
        kernel = value_2.args[1]
        conv_attrs = value_2.attrs

        x_ = relax.Var('x', x.struct_info)
        gamma_ = relax.Var('gamma', gamma.struct_info)
        beta_ = relax.Var('beta', beta.struct_info)
        moving_mean_ = relax.Var('moving_mean', moving_mean.struct_info)
        moving_var_ = relax.Var('moving_var', moving_var.struct_info)
        kernel_ = relax.Var('kernel', kernel.struct_info)
        
        cbr_attrs = dict(
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
        cbr_attrs = tvm.ir.make_node('DictAttrs', **cbr_attrs)

        bb = relax.BlockBuilder()
        with bb.function(fn_name, [x_, gamma_, beta_, moving_mean_, moving_var_, kernel_]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.nn.conv2d(x_, kernel_, **conv_attrs))
                lv1 = bb.emit(relax.op.nn.batch_norm(lv0, gamma_, beta_, moving_mean_, moving_var_, **bn_attrs))[0]
                lv2 = bb.emit(relax.op.nn.relu(lv1))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        

        fused_fn = bb.get()[fn_name].with_attr('Primitive', 1)
        return fused_fn, x, gamma, beta, moving_mean, moving_var, kernel, cbr_attrs


    def transform(self):
        # optimizer every relax function
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is not None and 'Primitive' in func.attrs and func['Primitive'] != 0:
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

        if not match_call(call, self.act):
            return call
        
        value_1 = self.lookup_binding(call.args[0])
        if isinstance(value_1, relax.TupleGetItem):
            value_1 = self.lookup_binding(value_1.tuple_value)
            
        if value_1 is None or not match_call(value_1, self.bn_op):
            return call
        
        value_2 = self.lookup_binding(value_1.args[0])
        if value_2 is None or not match_call(value_2, self.conv_op):
            return call
        
        fn_name = 'fused_cbr%d' % self.count
        self.count += 1
        fused_fn, x, gamma, beta, moving_mean, moving_var, kernel, cbr_attrs \
                    = self.create_cbr_op(call, value_1, value_2, fn_name)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        return relax.Call(global_var, [x, gamma, beta, moving_mean, moving_var, kernel], cbr_attrs)

@tvm.ir.transform.module_pass(opt_level=2, name="CBRFuse")
class FuseCBRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return ConvBRFusor(mod).transform()
