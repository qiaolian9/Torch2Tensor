import torch
import torchvision.models as models
import sys
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