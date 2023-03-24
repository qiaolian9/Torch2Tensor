import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from tabulate import tabulate
from tvm import relax, IRModule

torch.manual_seed(0)
np.random.seed(0)

from .t2t_relax import *
from .t2t_tir import *
from .t2t_torch.pytorch_graph import PytorchGraph
from .t2t_optimizer import mlc_tuner
from .benchmark import T2Tbenchmark
from .utils import (
    get_function_name
)


class T2TParser:
    def __init__(
        self, model, inputs, input_shapes, fuse=False, concrete_args=None, dynamic_batch=False, device_id = 0, **kwargs, 
    ):
        super(T2TParser, self).__init__()
        self.model = model.eval()
        self.inputs = inputs
        self.input_shapes = input_shapes
        if isinstance(self.inputs, torch.Tensor):
            self.inputs = [self.inputs]
        self.fuse = fuse
        self.concrete_args = concrete_args
        self.dynamic_batch = dynamic_batch
        self.mlc_tuner = mlc_tuner(**kwargs)
        self.benchmark = T2Tbenchmark(
            model, self.inputs, concrete_args, device_id, 
            kwargs['compile_tir_target'] if 'compile_tir_target' in kwargs else 'llvm')


    @staticmethod
    def print_tabular(graph_module):
        '''
            print computation graph(based on torch fx) 
        '''
        nodes = list(graph_module.graph.nodes)
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in nodes]
        logger.debug(
            tabulate(
                node_specs,
                headers=["\nopcode", "\nname", "\ntarget", "\nargs", "\nkwargs"],
            )
        )
    @staticmethod
    def print_ir(IR: IRModule):
        '''
            print tvm high/low level Tensor programs(include relax and TensorIR)
        '''
        logger.info(
            IR.show(black_format=False)
        )

    @staticmethod
    def print_op(IR: IRModule):
        fn_names = [x.name_hint for x in IR.functions]
        if 'main' in fn_names:
            fn_names.remove('main')
        logger.info(
            fn_names
        )

    ##### automatic code tune process #####
    def convert(self):
        '''
            convert torch.nn.Module to computation graph by using torch FX
            NOTE: only support static graph 
        '''
        if isinstance(self.model, GraphModule):
            pass
        elif isinstance(self.model, nn.Module):
            pass
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")
        
        self.pytorch_graph = PytorchGraph(
            self.model, self.inputs, self.concrete_args, self.dynamic_batch
        )
        self.print_tabular(self.pytorch_graph)
        self.state_dict = self.pytorch_graph.graph_module.state_dict()
        self.named_modules = dict(self.pytorch_graph.graph_module.named_modules())
        self.node_map = dict()
        self.gen_relax_graph()

    def gen_relax_graph(self):
        '''
            generate tvm relax IR from torch.fx.grapgModule
            NOTE: only support some ops in README now!
        '''
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
                        getattr_layer = GetAttrLayer(bb, node, self.node_map, module=self.model)
                        self.node_post_process(node, getattr_layer)
                    elif node.op == 'call_module':
                        module = self.named_modules[node.target]
                        if isinstance(module, (nn.Conv2d, )):
                            conv_layer = ConvLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, conv_layer)
                        elif isinstance(module, nn.BatchNorm2d):
                            batchnorm_layer = BatchNormLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, batchnorm_layer)
                        elif isinstance(module, nn.LayerNorm):
                            layernorm_layer = LayerNormLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, layernorm_layer)
                        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, 
                                                 nn.Hardswish, nn.Sigmoid, nn.Softmax,
                                                 nn.Hardsigmoid)):
                            activate_layer = ActivateLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, activate_layer)
                        elif isinstance(module, nn.Linear):
                            linear_layer = LinearLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, linear_layer)
                        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                            pool2d_layer = Pool2dLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, pool2d_layer)
                        elif isinstance(module, nn.AdaptiveAvgPool2d):
                            adaptiveavgpool2d_layer = Pool2dLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, adaptiveavgpool2d_layer)
                        elif isinstance(module, nn.Dropout):
                            dropout_layer = DropoutLayer(bb, node, self.node_map, module)
                            self.node_post_process(node, dropout_layer)
                        else:
                            logger.debug(node.target)
                            raise NotImplementedError('nn.Module type %s is not implemented now!' % type(module))
                    elif node.op == 'call_function':
                        function_name = get_function_name(node.target)
                        if function_name == 'add':
                            add_layer = AddFunc(bb, node, self.node_map)
                            self.node_post_process(node, add_layer)
                        elif function_name == 'sub':
                            sub_layer = SubFunc(bb, node, self.node_map)
                            self.node_post_process(node, sub_layer)
                        elif function_name == 'matmul':
                            matmul_layer = MatmulFunc(bb, node, self.node_map)
                            self.node_post_process(node, matmul_layer)
                        elif function_name == 'mul':
                            mul_layer = MulFunc(bb, node, self.node_map)
                            self.node_post_process(node, mul_layer)
                        elif function_name == 'floordiv':
                            floordiv_layer = FloorDivFunc(bb, node, self.node_map)
                            self.node_post_process(node, floordiv_layer)
                        elif function_name in ['relu', 'sigmoid', 'softmax']:
                            activate_layer = ActivateFunc(bb, node, self.node_map)
                            self.node_post_process(node, activate_layer)
                        elif function_name == 'flatten':
                            flatten_layer = FlattenFunc(bb, node, self.node_map)
                            self.node_post_process(node, flatten_layer)
                        elif function_name == 'concat' or function_name == 'cat':
                            concat_layer = ConcatFunc(bb, node, self.node_map)
                            self.node_post_process(node, concat_layer)
                        elif function_name == 'avg_pool2d':
                            avgpool2d_layer = Pool2dFunc(bb, node, self.node_map)
                            self.node_post_process(node, avgpool2d_layer)
                        elif function_name == 'boolean_dispatch': # F.maxpooling
                            avgpool2d_layer = Pool2dFunc(bb, node, self.node_map)
                            self.node_post_process(node, avgpool2d_layer)
                        elif function_name == 'adaptive_avg_pool2d': # F.maxpooling
                            adaptiveavgpool2d_layer = Pool2dFunc(bb, node, self.node_map)
                            self.node_post_process(node, adaptiveavgpool2d_layer)
                        elif function_name == 'transpose': # F.maxpooling
                            transpose_layer = TransposeFunc(bb, node, self.node_map)
                            self.node_post_process(node, transpose_layer)
                        elif function_name == 'getitem': # F.maxpooling
                            getitem_layer = GetItemFunc(bb, node, self.node_map)
                            self.node_post_process(node, getitem_layer)
                        elif function_name == 'getattr': # F.maxpooling
                            getattr_layer = GetAttrFunc(bb, node, self.node_map)
                            self.node_post_process(node, getattr_layer)
                        elif function_name == 'stochastic_depth': # F.maxpooling
                            stochastic_depth_layer = StochasticDepthFunc(bb, node, self.node_map)
                            self.node_post_process(node,stochastic_depth_layer)
                        else:
                            logger.debug(node.target)
                            raise NotImplementedError('func type %s is not implemented now!' % function_name)
                    elif node.op == 'call_method':
                        if str(node.target) == 'view':
                            reshape_layer = ReshapeFunc(bb, node, self.node_map)
                            self.node_post_process(node, reshape_layer)
                        elif str(node.target) == 'flatten':
                            flatten_layer = FlattenFunc(bb, node, self.node_map)
                            self.node_post_process(node, flatten_layer)
                        elif str(node.target) == 'size':
                            size_layer = SizeFunc(bb, node, self.node_map)
                            self.node_post_process(node, size_layer)
                        elif str(node.target) == 'contiguous':
                            contiguous_layer = ContiguousFunc(bb, node, self.node_map)
                            self.node_post_process(node, contiguous_layer)
                        elif str(node.target) == 'chunk':
                            chunk_layer = ChunkFunc(bb, node, self.node_map)
                            self.node_post_process(node, chunk_layer)
                        elif str(node.target) == 'mean':
                            mean_layer = MeanFunc(bb, node, self.node_map)
                            self.node_post_process(node, mean_layer)
                        else:
                            logger.debug(node.target)
                            raise NotImplementedError('method %s is not implemented now!' % node.target)
                    elif node.op == 'output':
                        if output_layer is not None:
                            raise Warning('output error')
                        output_layer = OutputLayer(bb, node, self.node_map)
                    else:
                        raise NotImplementedError("op type %s is not implemented" % (node.op))
            bb.emit_func_output(output_layer.value, fn_inputs)
        
        self.fn_inputs = fn_inputs
        self.RelaxIR = bb.get()
        
    def fuse_op(self):
        '''
            TO DO: op fused PASS
        '''
        from .t2t_optimizer.op_fuse import FuseCBRPass, FuseDenseAddPass, FuseCBPass
        self.RelaxIR = FuseDenseAddPass() (self.RelaxIR)
        # self.RelaxIR = FuseCBRPass()(self.RelaxIR)
        # self.RelaxIR = FuseCBPass()(self.RelaxIR)

    def gen_TensorIR(self):
        '''
            generate low level TensorIR from Relax IR
        '''
        from .t2t_tir.lower_to_tensorir_pass import LowerToTensorIRPass
        self.TensorIR = LowerToTensorIRPass()(self.RelaxIR)
        self.benchmark.TensorIR = self.TensorIR

    def tune_tir(self, op_list=None):
        if op_list is None:
            logger.info('tune all ops default')
            op_list = [x.name_hint for x in self.TensorIR.functions]
            if 'main' in op_list and isinstance(self.TensorIR['main'], relax.Function):
                op_list.remove('main')
        self.tuned_TensorIR = self.mlc_tuner.mlc_tune_tir(self.TensorIR, op_list)
        self.benchmark.TensorIR = self.tuned_TensorIR

    def check_result(self):
        self.benchmark.benchmark_init()
        self.benchmark.check_result()
    
    def infer_benchmark(self):
        self.benchmark.inf()

    ###### convert node map #####
    def node_post_process(self, node, relax_layer):
        '''
            used to record graph node & supported to generate Relax IR
        '''
        self.node_map[node] = relax_layer.value