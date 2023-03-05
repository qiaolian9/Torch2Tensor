import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from MLC.converter.pytorch_tvm_parser import PytorchRelaxParser, print_tensorIR
from test.model.vit import vit

# tv/m
torch.fx.wrap('len')
cls_model_name = ['alexnet', 'googlenet', 'vgg11', 'resnet50', 
                  'inception_v3', 'densenet121', 'mobilenet_v2', 
                  'shufflenet_v2_x1_0', 'regnet_y_400mf', 'mnasnet0_5', 
                  'squeezenet1_0', 'efficientnet_b0']

class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 30, 3, 1, 0, 1, 3, bias=False)
        self.linear = nn.Linear(3, 10)
        self.ada = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = len(x.size())
        return x
    
# CNN cls model
model = getattr(models, cls_model_name[0])()

# model = models.mobilenet_v3_small()
# model = vit()
# model = Demo()

x = torch.randn((1,3,224,224))
input_shapes = [(1,3,224,224)]

PR = PytorchRelaxParser(model, x, input_shapes)

PR.convert()
print_tensorIR(PR.relax_graph)
PR.gen_TensorIR()

# PR.mlc_tune_tir()

PR.print_op()

PR.check_result()