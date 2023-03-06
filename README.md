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
        from MLC.converter.pytorch_tvm_parser import PytorchRelaxParser, print_tensorIR

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
        print_tensorIR(PR.relax_graph) # -> show Relax IR
        PR.gen_TensorIR() # tvm relax IR -> tvm tensor IR
        #print_tensorIR(PR.TensorIR)  # show Tensor IR

        # PR.mlc_tune_tir() # auto schedule(ansor) tune tir

        PR.print_op() # print all operations in model
        PR.check_result() # check correct between tensor program and torch model
    ```

# Supported torch operations now(for high-level Relax IR)
|type|name|
|---|---|
|nn.Module|conv2d,batchnorm,relu,silu,relu6,linear,maxpool2d,adaptive_avg_pool2d,avg_pool2d,softmax,sigmoid,Dropout|
|function|flatten,add,relu,reshape,matmul,multiply,subtract,softmax,sigmoid,maxpool2d,avgpool2d,concat,transpose,floordiv,stochasticdepth|
|method|view(reshape),size,contiguous,chunk,mean,getitem,getattr|


# BenchMark
|task|type|name|
|---|---|---|
|Cls|CNN(12)|Alexnet,VGG11,Resnet50,Inceptionv3,GoogleNet,Densenet121,Mobilenetv2,Shufflenet,Regnet,MNasnet,Squeezenet1,EfficientNet|
|---|Transformer|ViT(*)|

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