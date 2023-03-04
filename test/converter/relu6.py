from tvm import relax, te
from tvm.script import relax as R
from tvm import topi
import tvm
import numpy as np
import torch

def relu6_te():
    x = te.placeholder((1000,1000),)
    t = topi.nn.relu(x)
    return te.compute(x.shape, lambda *i: te.min(t(*i), tvm.tir.const(6, t.dtype))), x

c, x = relu6_te()
mod = te.create_prim_func([c, x])
print(mod.show())
mod = tvm.build(mod, [c, x])

a = np.random.randn(1000, 1000).astype('float32')
a_torch = torch.from_numpy(a)
a_nd = tvm.nd.array(a)
b_nd = tvm.nd.empty((1000, 1000))

mod(b_nd, a_nd)
mod2 = torch.nn.ReLU6()
b_torch = mod2(a_torch)

np.testing.assert_allclose(b_torch.detach().numpy(), b_nd.numpy(), rtol=1e-5)
print(b_torch)
print(b_nd)
