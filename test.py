import torch
import torchvision.models as models
from Torch2Tensor.t2t import T2TParser

cls_model_name = ['alexnet', 'googlenet', 'vgg11', 'resnet50', 
                  'inception_v3', 'densenet121', 'mobilenet_v2', 
                  'shufflenet_v2_x1_0', 'regnet_y_400mf', 'mnasnet0_5', 
                  'squeezenet1_0', 'efficientnet_b0', 'mobilenet_v3_small']

mlc_dict = dict(target='cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152', work_dir="./tune_tmp", 
            task_name='main', max_trials_global=64, 
            num_trials_per_iter=32, compile_tir_target='cuda')


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class Demo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = conv3x3(3, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(3, 3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.downsample = None
        self.stride = 1
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    model = getattr(models, 'resnet18')()
    model = Demo()
    x = torch.randn((1,3, 224, 224))
    input_shapes = [(1,3, 224, 224)]

    PR = T2TParser(model, x, input_shapes, **mlc_dict)

    PR.convert()
    PR.print_tabular(PR.pytorch_graph)
    PR.print_ir(PR.RelaxIR)
    
    PR.fuse_op()
    PR.print_ir(PR.RelaxIR)

    PR.gen_TensorIR()
    PR.print_ir(PR.TensorIR)
    PR.print_op(PR.TensorIR)

    PR.tune_tir()
    PR.print_ir(PR.tuned_TensorIR)

    PR.check_result()
    PR.infer_benchmark()