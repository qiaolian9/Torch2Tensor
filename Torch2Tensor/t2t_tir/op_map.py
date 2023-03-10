from tvm import relax, topi
from .custom_te import *
from loguru import logger

DEFAULT_MAP_PATTERNS = dict()
    
def register_lower_pattern(te_map_name):
    def insert(fn):
        if te_map_name in DEFAULT_MAP_PATTERNS:
            raise Warning('te_map_name %s has been used' % te_map_name)
        DEFAULT_MAP_PATTERNS[te_map_name] = fn
        return fn
    
    return insert

def loggerdebug(x):
    logger.debug(x)
    logger.debug(type(x))

# relax op
@register_lower_pattern('relax.add')
def map_add_te(bb: relax.BlockBuilder, call: relax.Call):
    x, b = call.args
    return bb.call_te(topi.add, x, b)

@register_lower_pattern('relax.subtract')
def map_sub_te(bb: relax.BlockBuilder, call: relax.Call):
    x, b = call.args
    return bb.call_te(topi.subtract, x, b)

@register_lower_pattern('relax.matmul')
def map_matmul_te(bb: relax.BlockBuilder, call: relax.Call):
    x, w = call.args
    return bb.call_te(topi.nn.matmul, x, w)

@register_lower_pattern('relax.multiply')
def map_mul_te(bb:relax.BlockBuilder, call:relax.Call):
    x, y = call.args
    return bb.call_te(topi.multiply, x, y)

@register_lower_pattern('relax.floor_divide')
def map_mul_te(bb:relax.BlockBuilder, call:relax.Call):
    x, y = call.args
    return bb.call_te(topi.floor_divide, x, y)

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

@register_lower_pattern('relax.concat')
def map_concat_te(bb: relax.BlockBuilder, call: relax.Call):
    x = []
    for i in call.args[0]:
        x.append(i)
    attrs = call.attrs
    return bb.call_te(topi.concatenate, x, attrs.axis.value)

@register_lower_pattern('relax.chunk')
def map_chunk_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(chunk_te, x, attrs.chunks, attrs.dim)

# maybe wrong
# --------------------------------------
@register_lower_pattern('relax.shape_of')
def map_shape_of_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    return bb.call_te(topi.shape, x)

@register_lower_pattern('relax.permute_dims')
def map_permutedims_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    axes = call.attrs.axes
    if axes is not None:
        return bb.call_te(topi.transpose, x, axes)
    return bb.call_te(topi.transpose, x)

# @register_lower_pattern('relax.contiguous')
# def map_contiguous_te(bb: relax.BlockBuilder, call: relax.Call):
#     x = call.args[0]
#     return bb.call_te(contiguous_te, x)

# @register_lower_pattern('relax.get_item')
# def map_get_item_te(bb: relax.BlockBuilder, call: relax.Call):
#     x = call.args[0]
#     index = call.attrs.index
#     return bb.call_te(get_item_te, x, index)
# nn.op
@register_lower_pattern("relax.nn.softmax")
def map_softmax_te(bb: relax.BlockBuilder, call: relax.Call):
    x= call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.softmax, x, attrs.axis)

@register_lower_pattern("relax.sigmoid")
def map_sigmoid_te(bb: relax.BlockBuilder, call: relax.Call):
    x= call.args[0]
    return bb.call_te(topi.sigmoid, x)

@register_lower_pattern("relax.hardsigmoid")
def map_hardsigmoid_te(bb: relax.BlockBuilder, call: relax.Call):
    x= call.args[0]
    return bb.call_te(hardsigmoid_te, x)

@register_lower_pattern("relax.nn.dense")
def map_dense_te(bb: relax.BlockBuilder, call: relax.Call):
    x, kernel, bias = call.args
    return bb.call_te(dense_te, x, kernel, bias)

# nn.activation
@register_lower_pattern('relax.nn.relu')
def map_relu_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    return bb.call_te(topi.nn.relu, x)

@register_lower_pattern('relax.nn.silu')
def map_relu_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    return bb.call_te(silu_te, x)

@register_lower_pattern('relax.nn.relu6')
def map_relu6_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    return bb.call_te(relu6_te, x)

@register_lower_pattern('relax.nn.hardswish')
def map_hardswish_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    return bb.call_te(hardswish_te, x)

# nn.conv
@register_lower_pattern('relax.nn.conv2d')
def map_conv_te(bb: relax.BlockBuilder, call: relax.Call):
    x, k = call.args
    attrs = call.attrs
    if attrs.groups == 1:
        return bb.call_te(topi.nn.conv2d, x, k, attrs.strides, attrs.padding, attrs.dilation)
    else:
        return bb.call_te(topi.nn.group_conv2d_nchw, x, k, attrs.strides, attrs.padding, attrs.dilation, attrs.groups)

# nn.pooling
@register_lower_pattern('relax.nn.max_pool2d')
def map_maxpool2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.pool2d, x, attrs.pool_size, attrs.strides, attrs.dilation, attrs.padding, 'max', attrs.ceil_mode, attrs.layout)

@register_lower_pattern('relax.nn.avg_pool2d')
def map_avgpool2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.pool2d, x, attrs.pool_size, attrs.strides, attrs.dilation, attrs.padding, 'avg', attrs.ceil_mode, attrs.layout)

@register_lower_pattern('relax.nn.adaptive_avg_pool2d')
def map_adaptiveAvgPool2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.adaptive_pool, x, attrs.output_size, 'avg', attrs.layout)

# nn.norm
@register_lower_pattern('relax.nn.batch_norm')
def map_BatchNorm2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x, gamma, beta, moving_mean, moving_var = call.args[:5]
    attrs = call.attrs
    # batch_norm has 3 return values(value & moving mean/var)
    return bb.call_te(topi.nn.batch_norm, x, gamma, beta, moving_mean, moving_var, attrs.axis, attrs.epsilon, attrs.center, attrs.scale)

@register_lower_pattern('relax.nn.layer_norm')
def map_BatchNorm2d_te(bb: relax.BlockBuilder, call: relax.Call):
    x, gamma, beta = call.args[:3]
    attrs = call.attrs
    return bb.call_te(topi.nn.layer_norm, x, gamma, beta,  attrs.axes, attrs.epsilon)

# nn.dropout
@register_lower_pattern("relax.nn.dropout")
def map_dropout_te(bb: relax.BlockBuilder, call: relax.Call):
    x = call.args[0]
    rate = call.attrs.rate
    return bb.call_te(dropout_te, x, rate)