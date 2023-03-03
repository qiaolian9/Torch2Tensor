from tvm import relax, topi
from .custom_te import *

DEFAULT_MAP_PATTERNS = dict()
    
def register_lower_pattern(te_map_name):
    def insert(fn):
        if te_map_name in DEFAULT_MAP_PATTERNS:
            raise Warning('te_map_name %s has been used' % te_map_name)
        DEFAULT_MAP_PATTERNS[te_map_name] = fn
        return fn
    
    return insert

# relax op
@register_lower_pattern('relax.add')
def map_add_te(bb: relax.BlockBuilder, call: relax.Call):
    x, b = call.args
    return bb.call_te(topi.add, x, b)

@register_lower_pattern('relax.matmul')
def map_matmul_te(bb: relax.BlockBuilder, call: relax.Call):
    x, w = call.args
    return bb.call_te(topi.nn.matmul, x, w)

@register_lower_pattern('relax.variance')
def map_var_te(bb: relax.BlockBuilder, call: relax.Call):
    attrs = call.attrs
    x = call.args[0]
    return bb.call_te(var_te, x, attrs.axis)

@register_lower_pattern('relax.mean')
def map_mean_te(bb: relax.BlockBuilder, call: relax.Call):
    attrs = call.attrs
    x = call.args[0]
    return bb.call_te(mean_te, x, attrs.axis)

@register_lower_pattern('relax.reshape')
def map_reshape_te(bb: relax.BlockBuilder, call: relax.Call):
    x, shape = call.args
    return bb.call_te(topi.reshape, x, shape)

# nn.op
# @register_lower_pattern("relax.nn.dense")
# def map_dense_te(bb: relax.BlockBuilder, call: relax.Call):
#     x, w = call.args[0]
#     return bb.call_te(topi.nn.dense, x, w)

# nn.activation
@register_lower_pattern('relax.nn.relu')
def map_relu_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    return bb.call_te(topi.nn.relu, x)

# nn.conv
@register_lower_pattern('relax.nn.conv2d')
def map_conv_te(bb: relax.BlockBuilder, call: relax.Call):
    x, k = call.args
    attrs = call.attrs
    return bb.call_te(topi.nn.conv2d, x, k, attrs.strides, attrs.padding, attrs.dilation)

# nn.pooling
@register_lower_pattern('relax.nn.max_pool2d')
def map_maxpool2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.pool2d, x, attrs.pool_size, attrs.strides, attrs.dilation, attrs.padding, 'max', attrs.ceil_mode, attrs.layout)

@register_lower_pattern('relax.nn.adaptive_avg_pool2d')
def map_adaptiveAvgPool2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.adaptive_pool, x, attrs.output_size, 'avg', attrs.layout)

# nn.norm
def map_BatchNorm2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x, gamma, beta, moving_mean, moving_var = call.args[:5]
    attrs = call.attrs
    # batch_norm has 3 return values(value & moving mean/var)
    return bb.call_te(topi.nn.batch_norm, x, gamma, beta, moving_mean, moving_var, attrs.axis, attrs.epsilon, attrs.center, attrs.scale)[0]
