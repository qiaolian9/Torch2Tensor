import torch
import torchvision.models as models
import torch.nn.functional as F
from MLC.converter.pytorch_tvm_parser import PytorchRelaxParser
# tv/m

class Demo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 10, 3, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(3, 10)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = Demo()
model = models.resnet18()
x = torch.rand(1,3,224,224)
input_shapes = [(1,3,224,224)]

PR = PytorchRelaxParser(model, x, input_shapes)

PR.convert()
PR.print_tabular(PR.pytorch_graph)
PR.print_tensorIR(PR.relax_graph)
PR.gen_TensorIR()
# PR.print_tensorIR(PR.TensorIR)

PR.print_op()

PR.check_result()

from tvm import relax
# relax.op.permute_dims()
relax.const