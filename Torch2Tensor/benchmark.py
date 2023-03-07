from loguru import logger
import torch
import tvm
from tvm import relax
import numpy as np
from .utils import (
    map_reduce,
    gen_numpy_data,
    gen_tvm_data,
    gen_torch_data
)

class T2Tbenchmark:
    def __init__(self, model, inputs, concrete_args=None, device_id=0, device='llvm', tensorIR=None) -> None:
        self.TensorIR = tensorIR
        self.concrete_args = concrete_args
        self.device_id = device_id
        self.device = device
        self.n_repeat = 1000
        self.model = model
        self.inputs = inputs
        if self.device == 'llvm':
            self.dev = tvm.cpu(self.device_id)
            self.dev_torch = torch.device('cpu')
        elif self.device == 'cuda':
            self.dev = tvm.cuda(self.device_id)
            self.dev_torch = torch.device('cuda:%d' % self.device_id)
        else:
            raise NotImplementedError("target device %s is not supported!" % self.device)
        
        
    ###### check result between torch & TVM #####
    def benchmark_init(self):
        # model init
        self.model = self.model.eval()
        if self.device == 'cuda':
            self.model = self.model.to(self.dev_torch)

        # tvm data
        self.tvm_input_list = map_reduce(self.inputs, gen_tvm_data, self.dev)
        if self.concrete_args is not None:
            self.tvm_input_list.extend(map_reduce(*self.concrete_args), gen_tvm_data, self.dev)
            
        # torch data
        self.torch_input_list = map_reduce(self.inputs, gen_torch_data, self.dev_torch)
        if self.concrete_args is not None:
            self.torch_input_list.extend(map_reduce(*self.concrete_args), gen_torch_data, self.dev_torch)

    def check_result(self):
        '''
            check result between torch and tvm
        '''
        self.pyotrch_inference()
        self.TensorIR_inference()

        pytorch_output_list = map_reduce(self.pytorch_output, gen_numpy_data)
        tvm_output = map_reduce(self.TensorIR_output, gen_numpy_data)     

        assert len(pytorch_output_list) == len(
            tvm_output
        ), "pytorch_output: %d vs tvm_output %d" % (
            len(pytorch_output_list),
            len(tvm_output),
        )
        
        for idx in range(len(tvm_output)):
            np.testing.assert_allclose(
                pytorch_output_list[idx],
                tvm_output[idx],
                rtol=5e-2,
                atol=1e-5
            )
        logger.info("accuracy test passed")

    def pyotrch_inference(self):
        with torch.no_grad():
            self.pytorch_output = self.model(*self.torch_input_list)

        if isinstance(self.pytorch_output, torch.Tensor):
            self.pytorch_output = [self.pytorch_output]
        
    def TensorIR_inference(self):
        ex = relax.vm.build(self.TensorIR, self.device)

        self.vm = relax.vm.VirtualMachine(ex, self.dev)
        self.TensorIR_output = []
        for i in self.tvm_input_list:
            self.TensorIR_output.append(self.vm['main'](i))

    ##### inf performance #####
    def inf(self):
        f_timer_mod = self.vm.time_evaluator('main', self.dev, number=self.n_repeat)
        tensor_mod_time = []

        for i in self.tvm_input_list:
            tensor_mod_time.append((f_timer_mod(i).mean * 1e3))
            # break
        logger.info("tensor program inf time: %f(ms)" % np.mean(tensor_mod_time))
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(50):
            _ = self.model(*self.torch_input_list)
        end.record()
        torch.cuda.synchronize()

        timings = []
        with torch.no_grad():
            for i in self.torch_input_list:
                for _ in range(self.n_repeat):
                    start.record()
                    _ = self.model(i)
                    end.record()
                    torch.cuda.synchronize()
                    curr_timing = start.elapsed_time(end)
                    timings.append(round(curr_timing, 5))
                # break

        logger.info("torch model inf time : %f(ms)" % (np.mean(timings)))