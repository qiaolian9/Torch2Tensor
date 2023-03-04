from tvm import relax, te
from tvm.script import relax as R
from tvm import topi
import tvm
import numpy as np
import torch

def relu6_te():
    x = te.placeholder((3, 30, 224, 224),)
    k = te.placeholder((30, 1, 3, 3))
    t = topi.nn.group_conv2d_nchw(x, k, 1, 1, 1, 3)
    return t, x, k

c, x, k = relu6_te()
mod = te.create_prim_func([c, x, k])
print(mod.show())
mod = tvm.build(mod, [c, x, k])

a = np.random.randn(3, 30, 224, 224).astype('float32')
k = np.random.randn(30, 1, 3, 3).astype('float32')

a_nd = tvm.nd.array(a)
b_nd = tvm.nd.empty((3, 30, 224, 224))
k_nd = tvm.nd.array(k)

mod(b_nd, a_nd, k_nd)
