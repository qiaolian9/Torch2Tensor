# Torch2Tensor
**Torch nn.Module** -> **TVM Tensor program**
## process
```
nn.Module ---> fx.graph ---> tvm relax IR ---> tvm tensor IR ---> tuned tensor IR ---> ex&vm
torch fx based pytorch model compiler, including relax ---> low level TensorIR
```
# Installation
```bash
pip install git+git@github.com:qiaolian9/mlc.git
```

# How to use

* torch2tvm
    ``` python
        import torch
        import torchvision.models as models
        import torch.nn.functional as F
        from torch import nn
        from MLC.converter.pytorch_tvm_parser import PytorchRelaxParser, print_tensorIR
        from test.model.vit import vit

        cls_model_name = ['alexnet', 'googlenet', 'vgg11', 'resnet50', 
                  'inception_v3', 'densenet121', 'mobilenet_v2', 
                  'shufflenet_v2_x1_0', 'regnet_y_400mf', 'mnasnet0_5', 
                  'squeezenet1_0', 'efficientnet_b0']

        # CNN cls model
        model = getattr(models, cls_model_name[0])()

        x = torch.randn((1,3,224,224))
        input_shapes = [(1,3,224,224)]

        PR = PytorchRelaxParser(model, x, input_shapes)
        PR.convert() # torch.nn.Module -> torch.fx.graph & tvm relax IR
        print_tensorIR(PR.relax_graph)
        PR.gen_TensorIR() # tvm relax IR -> tvm tensor IR

        # PR.mlc_tune_tir() # auto schedule(ansor) tune tir

        PR.print_op() # print all operations in model
        PR.check_result() # check correct between tensor program and torch model
    ```

# supported torch operation(for high-level Relax IR)
|type|name|
|---|---|
|nn.Module|conv2d,batchnorm,relu,silu,relu6,linear,maxpool2d,adaptive_avg_pool2d,avg_pool2d,softmax,sigmoid,Dropout|
|function|flatten,add,relu,reshape,matmul,multiply,subtract,softmax,sigmoid,maxpool2d,avgpool2d,concat,transpose,floordiv,stochasticdepth|
|method|view(reshape),size,contiguous,chunk,mean,getitem,getattr|


# BenchMark
|task|type|name|
|---|---|---|
|Cls|CNN(10)|Alexnet,Resnet50,Inceptionv3,GoogleNet,Densenet121,Mobilenetv2,Regnet,MNasnet,Squeezenet1,EfficientNet|
|---|Transformer|ViT(*)|


# Torch FX & TVM Relax
1. graph tracer based on an open source project <https://github.com/inisis/brocolli> \
2. Relax based on project<https://github.com/tlc-pack/relax>

# To Do
1. register xxattrs(transpose & avgpool2d)
2. torch.fx.wrap
3. fix getitem potential bug
