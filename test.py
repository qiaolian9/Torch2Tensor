import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from MLC.converter.pytorch_tvm_parser import PytorchRelaxParser
from torchvision.ops.misc import ConvNormActivation
from typing import Callable, Any, Optional, List
# tv/m

class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 30, 3, 1, 0, 1, 3, bias=False)

    def forward(self, x):
        # x = self.conv(x)
        x = x.view(3, 224, 224)
        # x = x.view(1, 3, 224, 224)
        return x
# model = models.alexnet()
# model = models.googlenet()
# model = models.resnet50()
# model = models.inception_v3()
# model = models.densenet121()
# model = models.mobilenet_v2()

model = models.efficientnet_b2()
model = models.shufflenet_v2_x1_0()

# model = Demo()

x = torch.randn((1,3,224,224))
input_shapes = [(1,3,224,224)]

PR = PytorchRelaxParser(model, x, input_shapes)

PR.convert()
# PR.print_tensorIR(PR.relax_graph)
PR.gen_TensorIR()
# PR.print_tensorIR(PR.TensorIR)

PR.print_op()

PR.check_result()

from tvm import relax
# relax.op.permute_dims()
torch.nn.Dropout()