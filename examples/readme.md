# Examples
**More details in example.ipynb, it includes all the IRModue code**
## 1.import pakage
```python
import torch
import torchvision.models as models
from Torch2Tensor.t2t import T2TParser
```
## 2.load your model
```python
# test demo
class Demo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.ada = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.ada(x).view((1, -1))
        x = self.linear(x)
        return x

# model = getattr(models, 'alexnet')()
model = Demo()
x = torch.randn((1,3,224, 224))
input_shapes = [(1,3, 224, 224)]
```
## 3.define tund config
```python
# note: if you only have cpu cpu platform, ignore this dict 
mlc_dict = dict(target='cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152', work_dir="./demo", 
            task_name='main', max_trials_global=32, 
            num_trials_per_iter=32, compile_tir_target='cuda')
```

## 4.torch -> relax
```python
# PR = T2TParser(model, x, input_shapes) default fot llvm
PR = T2TParser(model, x, input_shapes, device_id=0, **mlc_dict)
PR.convert()
PR.print_tabular(PR.pytorch_graph)
PR.print_ir(PR.relax_graph)
```
## 5.relax graph optimizer
to be done

## 6.relax -> tensor
```python
PR.gen_TensorIR()
PR.print_ir(PR.TensorIR)
PR.print_op(PR.TensorIR)
```

## 7.automatic tensor code optimizer
```python
# tune for gpu platform
PR.tune_tir()
PR.print_ir(PR.tuned_TensorIR)
```

## 8.performance test
```python
PR.check_result()
PR.infer_benchmark()
```