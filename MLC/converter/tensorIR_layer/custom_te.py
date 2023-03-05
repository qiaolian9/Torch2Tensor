from tvm import te, relax
from tvm import topi
import numpy as np
import tvm
from loguru import logger

__all__ = ['mean_te', 'var_te', 'silu_te', 'dropout_te', 'relu6_te', 'contiguous_te', 'chunk_te', 'get_item_te']

def loggerdebug(x):
    logger.debug(x)
    logger.debug(type(x))

def mean_te(x, axis):
    shape = x.shape
    out_shape = []
    ori_axis = []
    reduce_axis = []
    mean_ = 1
    for i, ax in enumerate(shape):
        if i in axis:
            reduce_axis.append(te.reduce_axis((0, shape[i])))
            mean_ *= shape[i]
        else:
            out_shape.append(ax.value)
            ori_axis.append(i)   

    mean_ = te.const(int(mean_))
    def fn(*i):
        k = []
        num_a = 0
        num_b = 0
        for j in range(len(shape)):
            if j in axis:
                k.append(reduce_axis[num_b])
                num_b += 1
            else:
                k.append(i[num_a])
                num_a += 1
        return te.sum(x(*tuple(k)), axis=reduce_axis)
    sum_ = te.compute(out_shape, fn, name='sum_out')
    out = te.compute(out_shape, lambda *i: sum_(*i) / mean_, name='mean_out')
    return out

def var_te(x, axis):
    shape = x.shape
    reduce_axis = []
    n_ = 1
    for i in axis:
        reduce_axis.append(te.reduce_axis((0, shape[int(i)])))
        n_ *= shape[int(i)]
    n_ = te.const(int(n_))
    mean_ = mean_te(x, axis)
    sub_ = te.compute(x.shape, lambda n, c, h, w: te.power(x[n, c, h, w] - mean_[c], 2), name='sub_out')
    sum_ = te.compute(mean_.shape, lambda i: te.sum(sub_[reduce_axis[0],i,reduce_axis[1],reduce_axis[2]], axis=reduce_axis), name='sum_out')
    var_out = te.compute(sum_.shape, lambda i: sum_[i] / n_, name='var_out')
    return var_out

# nn.activate
def silu_te(x):
    t = topi.sigmoid(x)
    return topi.multiply(x, t)

def relu6_te(x):
    t = topi.nn.relu(x)
    return te.compute(x.shape, lambda *i: te.min(t(*i), tvm.tir.const(6, t.dtype)))

def dropout_te(x, rate):
    out = te.compute(x.shape, lambda *i: x(*i))
    mask = te.compute(x.shape, lambda *i: x(*i))
    return [out, mask]

def contiguous_te(x):
    return te.compute(x.shape, lambda *i: x(*i))

def chunk_te(x, chunks, dim=0):
    data_shape = x.shape
    out_shape = []
    for i in data_shape:
        out_shape.append(i.value)
    axis = data_shape[dim.value]
    out = []
    if axis % chunks ==  0:
        num = axis // chunks
        out_shape[dim.value] = num.value
        for chunk_index in range(chunks.value):
            index = lambda *j:[k if step != dim else k + chunk_index * num for step, k in enumerate(j)]
            out_chunl = te.compute(out_shape, lambda *i: x(*(index(*i))))
            out.append(out_chunl)
    else:
        raise Warning('to be developed')
    return out

def get_item_te(x, index):
    return te.compute((1,), lambda i: x[0])
