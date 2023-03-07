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

class Demo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.ada = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(3, 1)
    
    def forward(self, x):
        # x = self.conv(x)
        x = self.ada(x)
        # x = x.view((1, -1))
        # x = self.linear(x)
        return x

if __name__ == "__main__":
    model = getattr(models, 'efficientnet_b0')()
    # model = Demo()
    x = torch.randn((1,3,224, 224))
    input_shapes = [(1,3, 224, 224)]

    PR = T2TParser(model, x, input_shapes, **mlc_dict)

    PR.convert()
    PR.print_tabular(PR.pytorch_graph)
    PR.print_ir(PR.relax_graph)
    
    PR.gen_TensorIR()
    PR.print_ir(PR.TensorIR)
    PR.print_op(PR.TensorIR)

    PR.tune_tir()
    PR.print_ir(PR.tuned_TensorIR)

    PR.check_result()
    PR.infer_benchmark()