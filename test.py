import torch
import torchvision.models as models
import torch.nn.functional as F
from MLC.converter.pytorch_tvm_parser import PytorchRelaxParser



class Demo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.conv = torch.nn.AvgPool2d(2,1)
    
    def forward(self, x):
        # x = self.conv(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x

model = Demo()
model = models.resnet18()
x = torch.rand(1,3,224,224)
input_shapes = [(1,3,224,224)]

PR = PytorchRelaxParser(model, x, input_shapes)

PR.convert()
PR.print_tabular(PR.pytorch_graph)
PR.print_tensorIR(PR.relax_graph)
# PR.gen_TensorIR()
# PR.print_tensorIR(PR.TensorIR)

from tvm import relax
import tvm
