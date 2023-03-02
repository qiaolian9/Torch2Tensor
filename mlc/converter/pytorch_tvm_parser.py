import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from tabulate import tabulate
from tvm import relax
import tvm

torch.manual_seed(0)
np.random.seed(0)

from .relax_layers import *
from .tensorIR_layer import *
from .pytorch_graph import PytorchGraph
from .common_utils import (
    map_reduce,
    gen_numpy_data,
    gen_tvm_data
)


class PytorchRelaxParser:
    def __init__(
        self, model, inputs, input_shapes, fuse=False, concrete_args=None, dynamic_batch=False, device='cpu'
    ):
        super(PytorchRelaxParser, self).__init__()
        self.model = model.eval()
        self.inputs = inputs
        self.input_shapes = input_shapes
        if isinstance(self.inputs, torch.Tensor):
            self.inputs = [self.inputs]
        self.fuse = fuse
        self.concrete_args = concrete_args
        self.dynamic_batch = dynamic_batch
        self.device = device

    def print_tabular(self, graph_module):
        nodes = list(graph_module.graph.nodes)
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in nodes]
        logger.debug(
            tabulate(
                node_specs,
                headers=["\nopcode", "\nname", "\ntarget", "\nargs", "\nkwargs"],
            )
        )

    def print_tensorIR(self, tensorIR):
        logger.debug(
            tensorIR.show()
        )

    def convert(self):
        if isinstance(self.model, GraphModule):
            pass
        elif isinstance(self.model, nn.Module):
            pass
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        self.pytorch_graph = PytorchGraph(
            self.model, self.inputs, self.concrete_args, self.dynamic_batch
        )
        self.state_dict = self.pytorch_graph.graph_module.state_dict()
        self.named_modules = dict(self.pytorch_graph.graph_module.named_modules())
        self.node_map = dict()
        self.gen_relax_graph()

    def gen_relax_graph(self):
        input_index = 0
        fn_inputs = []
        bb = relax.BlockBuilder()
        output_layer = None
        with bb.function('main'):
            with bb.dataflow():
                for node in self.pytorch_graph.nodes:
                    if node.op == 'placeholder':
                        input_shape = self.input_shapes[input_index]
                        input_index += 1
                        input_layer = InputLayer(bb, node, input_shape)
                        fn_inputs.append(input_layer.value)
                        self.node_post_process(node, input_layer)
                    elif node.op == 'get_attr':
                        pass
                    elif node.op == 'call_module':
                        module = self.named_modules[node.target]
                        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                            conv_layer = ConvLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, conv_layer)
                    elif node.op == 'call_function':
                        pass
                    elif node.op == 'call_method':
                        pass
                    elif node.op == 'output':
                        if output_layer is not None:
                            raise Warning('output error')
                        output_layer = OutputLayer(bb, node, self.node_map)
                    else:
                        raise NotImplementedError("op type %s is not implemented" % (node.op))
            bb.emit_func_output(output_layer.value, fn_inputs)
        
        self.fn_inputs = fn_inputs
        self.relax_graph = bb.get()
    
    def fuse_op(self):
        pass

    def gen_TensorIR(self):
        from .tensorIR_layer.lower_to_tensorir_pass import LowerToTensorIRPass
        self.TensorIR = LowerToTensorIRPass()(self.relax_graph)
   
    def check_result(self):
        self.pyotrch_inference()
        self.onnx_inference()
        pytorch_output_list = map_reduce(self.pytorch_output, gen_numpy_data)

        assert len(pytorch_output_list) == len(
            self.tvm_output
        ), "pytorch_output: %d vs tvm_output %d" % (
            len(pytorch_output_list),
            len(self.tvm_output),
        )
        
        for idx in range(len(self.onnx_output)):
            np.testing.assert_allclose(
                pytorch_output_list[idx],
                self.tvm_output[idx],
                rtol=1e-7,
                atol=1e-3,
            )
        logger.info("accuracy test passed")

    def pyotrch_inference(self):
        with torch.no_grad():
            if self.concrete_args is not None:
                self.pytorch_output = self.model(*self.inputs, **self.concrete_args)
            else:
                self.pytorch_output = self.model(*self.inputs)

        if isinstance(self.pytorch_output, torch.Tensor):
            self.pytorch_output = [self.pytorch_output]

    def get_Tensor_input(self):
        np_data = map_reduce(self.inputs, gen_numpy_data)
        self.tensor_input_list = map_reduce(np_data, gen_tvm_data)
        if self.concrete_args is not None:
            self.tensor_input_list.extend(map_reduce(*self.concrete_args), gen_tvm_data)
        

    def TensorIR_inference(self):
        if self.device == 'cpu':
            device = tvm.cpu()
        elif self.device == 'cuda':
            device = tvm.cuda()
        else:
            raise NotImplementedError("target device %s is not supported!" % self.device)
        ex = relax.vm.build(self.TensorIR, device)
        vm = relax.vm.VirtualMachine(ex, device)
        self.TensorIR_output = [vm['main'](self.tensor_input_list)]


    def node_post_process(self, node, relax_layer):
        self.node_map[node] = relax_layer.value