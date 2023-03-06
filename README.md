# Torch2Tensor

**Torch nn.Module** -> **TVM Tensor program**

A easy tool for generating Tensor Program from torch nn.module

## Main Process
```
nn.Module ---> fx.graph ---> tvm relax IR ---> tvm tensor IR ---> tuned tensor IR ---> ex&vm
```

# How to use
* Torch2Tensor
    ``` python
        import torch
        import torchvision.models as models
        from Torch2Tensor.converter.pytorch_tvm_parser import PytorchRelaxParser, print_tensorIR

        cls_model_name = ['alexnet', 'googlenet', 'vgg11', 'resnet50', 
                  'inception_v3', 'densenet121', 'mobilenet_v2', 
                  'shufflenet_v2_x1_0', 'regnet_y_400mf', 'mnasnet0_5', 
                  'squeezenet1_0', 'efficientnet_b0', 'mobilenet_v3_small]
        

        # CNN cls model
        model = getattr(models, cls_model_name[0])()

        x = torch.randn((1,3,224,224))
        input_shapes = [(1,3,224,224)]

        PR = PytorchRelaxParser(model, x, input_shapes)
        PR.convert() # torch.nn.Module -> torch.fx.graph & tvm relax IR
        print_tensorIR(PR.relax_graph) # -> show Relax IR
        PR.gen_TensorIR() # tvm relax IR -> tvm tensor IR
        # print_tensorIR(PR.TensorIR)  # show Tensor IR

        # PR.mlc_tune_tir() # auto schedule(ansor) tune tir

        PR.print_op() # print all operations in model
        PR.check_result() # check correct between tensor program and torch model
    ```

# Supported torch operations now(for high-level Relax IR)
|type|name|
|---|---|
|nn.Module|Conv2d, BatchNorm, LayerNorm, Linear/Dense, Maxpool2d, AdaptiveAvgPool2d,<br>Avgpool2d, Softmax, Sigmoid, ReLU, SiLU, ReLU6, Hardsigmoid, Hardswish, Dropout|
|function|flatten, add, relu, reshape, matmul, multiply, subtract,softmax, sigmoid, <br>maxpool2d, avgpool2d, concat, transpose,floordiv, stochasticdepth|
|method|view(reshape), size, contiguous, chunk, mean, getitem, getattr|


# Supported BenchMark
|task|type|name|
|---|---|---|
|Cls|CNN(13)|Alexnet,VGG11,Resnet50,Inceptionv3,GoogleNet,Densenet121,Mobilenetv2,<br>Shufflenet,Regnet,MNasnet,Squeezenet,EfficientNet,MobileNetv3|
|---|Transformer|SimpleViT,ViT(*)|

# Installation
```bash
# still in developing, maybe unstable
pip install git+git@github.com:qiaolian9/mlc.git
```

# Acknowledgement
1. [Relax](https://github.com/tlc-pack/relax): relax(relay next)
2. [brocolli](https://github.com/inisis/brocolli): Torch Fx Pytorch Model Converter(for onnx & caffe)
3. [MLC](https://mlc.ai/summer22-zh/): Machine Learning Compiler(chen etc.)

# To Do
1. register xxattrs(transpose & avgpool2d)
2. torch.fx.wrap (eg. func-len())
3. fix getitem potential bug
4. develop more ops

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
1. register your op in dir: converter/register_relax/
2. construct your op_layer in dir: converter/relax_layer/
3. define your low-level te computation in dir: converter/tensorIR_layer/
```