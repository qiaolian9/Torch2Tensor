# mlc
Torch FX module convertor -> TVM

torch fx based pytorch model converter, including pytorch2tvm_relax
torch fx based pytorch model compiler, including relax ---> low level TensorIR

# Installation
```bash
pip install git+git@github.com:qiaolian9/mlc.git
```

# How to use

* torch2tvm
    ``` python
    import torch
    import torchvision.models as models
    from mlc.converter.pytorch_tvm_parser import PytorchRelaxParser

    class Demo(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
        
        def forward(self, x):
            x = self.conv(x)
            return x

    model = Demo()
    # model = models.resnet18()
    x = torch.rand(1,3,224,224)
    input_shapes = [(1,3,224,224)]

    PR = PytorchRelaxParser(model, x, input_shapes)
    PR.convert()
    PR.print_tabular(PR.pytorch_graph)
    PR.print_tensorIR(PR.relax_graph)
    PR.gen_TensorIR()
    PR.print_tensorIR(PR.TensorIR)
    ```
# Torch FX tools
graph tracer based on an open source project <https://github.com/inisis/brocolli>

# TVM Relax(Relay next)
Relax based on project<https://github.com/tlc-pack/relax>

# supported torch operation(for high-level Relax IR)
|type|name|
|---|---|
|nn.Module|conv2d,batchnorm,relu,silu,linear,maxpool2d,adaptive_avg_pool2d,softmax,sigmoid|
|function|flatten,add,relu,reshape,matmul,subtract,softmax,sigmoid|
|method|view(reshape)|

# Tensor Expression operation(for low-level Tensor IR)
|type|name|
|---|---|
|relax.op|add,subtract,matmul,variance,mean,reshape,permute_dims|
|relax.nn|softmax,sigmoid,relu,silu,conv2d,maxpool2d,adaptiveavgpool2d,batchnorm|

# main API
```python
PR = PytorchRelaxParser(model, x, input_shapes)
... details in  class PytorchRelaxParser
```