# Torch2Tensor

**Torch nn.Module** -> **TVM Tensor program**

A easy tool for generating Tensor Program from torch nn.module

## Main Process
```
nn.Module ---> fx.graph ---> tvm relax IR ---> graph fused relax IR ---> tvm tensor IR ---> tuned tensor IR ---> ex&vm
```

# How to use
**Usage Detail in [/example/example.ipynb](https://github.com/qiaolian9/Torch2Tensor/tree/main/examples/example.ipynb)**
* Torch2Tensor
    ``` python
        import torch
        import torchvision.models as models
        from Torch2Tensor.t2t import T2TParser

        cls_model_name = ['alexnet', 'googlenet', 'vgg11', 'resnet50', 
                        'inception_v3', 'densenet121', 'mobilenet_v2', 
                        'shufflenet_v2_x1_0', 'regnet_y_400mf', 'mnasnet0_5', 
                        'squeezenet1_0', 'efficientnet_b0', 'mobilenet_v3_small']

        # CNN cls model
        mlc_dict = dict(target='cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152', work_dir="./tune_tmp", 
                    task_name='main', max_trials_global=64, 
                    num_trials_per_iter=32, compile_tir_target='cuda')
        
        class Demo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.ada = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.linear = torch.nn.Linear(3, 1)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.ada(x).view((1, -1))
                x = self.linear(x)
                return x

        if __name__ == "__main__":
            model = Demo()
            # model = getattr(models, 'resnet50')()
            x = torch.randn((1,3,224, 224))
            input_shapes = [(1,3, 224, 224)]

            PR = T2TParser(model, x, input_shapes, **mlc_dict)

            PR.convert()      # torch.nn.Module -> torch.fx.graph & tvm relax IR
            PR.print_tabular(PR.pytorch_graph)
            PR.print_ir(PR.relax_graph)   # -> show Relax IR
            
            PR.gen_TensorIR()   # tvm relax IR -> tvm tensor IR
            PR.print_ir(PR.TensorIR)   # show Tensor IR
            PR.print_op(PR.TensorIR)   # print all operations in model

            PR.tune_tir()  # ansor: automatic tensor code optimizer
            PR.print_ir(PR.tuned_TensorIR)  # show optimizer tensor program

            PR.check_result()  # check tensor program correctness
            PR.infer_benchmark()  # test tensor program performance
    ```
# Supported torch operations now(for high-level Relax IR)
|type|name|
|---|---|
|nn.Module|Conv2d, BatchNorm, LayerNorm, Linear/Dense, Maxpool2d, AdaptiveAvgPool2d, Avgpool2d, Softmax, Sigmoid, ReLU, SiLU, ReLU6, Hardsigmoid, Hardswish, Dropout|
|function|flatten, add, relu, reshape, matmul, multiply, subtract, softmax, sigmoid, maxpool2d, avgpool2d, concat, transpose, floordiv, stochasticdepth|
|method|view(reshape), size, contiguous, chunk, mean, shape, getitem, getattr|


# Supported BenchMark
|task|type|name|
|---|---|---|
|Cls|CNN(13)|Alexnet,VGG11,Resnet50,Inceptionv3,GoogleNet,Densenet121,Mobilenetv2,<br>Shufflenet,Regnet,MNasnet,Squeezenet,EfficientNet,MobileNetv3|
|---|Transformer|SimpleViT,ViT(*)|

<!-- # Installation
```bash
# still in developing, maybe unstable
pip install git+git@github.com:qiaolian9/mlc.git 
```-->

# Acknowledgement
1. [Relax](https://github.com/tlc-pack/relax): relax(relay next)
2. [brocolli](https://github.com/inisis/brocolli): Torch Fx Pytorch Model Converter(for onnx & caffe)
3. [MLC](https://mlc.ai/summer22-zh/): Machine Learning Compiler(chen etc.)

# To Do
1. support more ops
2. add user own graph optimizer pass
3. complete CV:cls models benchmark test 

# Note
1. register op nn.dense now support input data with more than 2-dim tensor
2. if you use rearrange from einops, you should use primitive op(transpose, view, contiguous,etc.) replace it.(due to func len in einops)
```python
# out = rearrange(out, 'b h n d -> b n (h d)')
out = torch.transpose(out, 1, 2).contiguous()
shape = out.size()
out = out.view((shape[0], shape[1], -1))
```
3. if you wanna register your own special op, you should do 3 steps:
```
1. register your op in dir: Torch2Tensor/t2t_relax/op
2. construct your op_layer in dir: /Torch2Tensor/t2t_relax/module or function
3. define your own graph fuse pattern in dir: Torch2Tensor/t2t_optimizer/op_fuse
4. define your low-level te computation in dir: Torch2Tensor/t2t_tir
```
4. **still in developing, maybe unstable**