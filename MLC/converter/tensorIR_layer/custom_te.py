from tvm import te

__all__ = ['mean_te', 'var_te']

def mean_te(x, axis):
    shape = x.shape
    reduce_axis = []
    mean_ = 1
    for i in axis:
        reduce_axis.append(te.reduce_axis((0, shape[int(i)])))
        mean_ *= shape[int(i)]
    mean_ = te.const(int(mean_))
    sum_ = te.compute((int(shape[1]),), lambda i: te.sum(x[reduce_axis[0],i,reduce_axis[1],reduce_axis[2]], axis=reduce_axis), name='sum_out')
    out = te.compute(sum_.shape, lambda *i: sum_(*i) / mean_, name='mean_out')
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