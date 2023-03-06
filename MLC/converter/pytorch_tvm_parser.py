import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from tabulate import tabulate
from tvm import relax, IRModule
import tvm

# torch.manual_seed(0)
# np.random.seed(0)

from .relax_layers import *
from .tensorIR_layers import *
from .pytorch_graph import PytorchGraph
from .common_utils import (
    map_reduce,
    gen_numpy_data,
    gen_tvm_data,
    get_function_name
)


def print_tensorIR(TensorIR):
    '''
        print tvm high/low level Tensor programs(include relax and TensorIR)
    '''
    logger.info(
        TensorIR.show()
    )

class PytorchRelaxParser:
    def __init__(
        self, model, inputs, input_shapes, fuse=False, concrete_args=None, dynamic_batch=False, device='llvm',
        target='cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152', 
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
        self.target = target

    def print_tabular(self, graph_module):
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

    def print_op(self):
        fn_names = [x.name_hint for x in self.TensorIR.functions]
        if 'main' in fn_names:
            fn_names.remove('main')
        logger.info(
            fn_names
        )

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
                        getattr_layer = GetAttrLayer(bb, node, module=self.model)
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
                        elif function_name == 'relu':
                            relu_layer = ReluFunc(bb, node, self.node_map)
                            self.node_post_process(node, relu_layer)
                        elif function_name == 'flatten':
                            flatten_layer = FlattenFunc(bb, node, self.node_map)
                            self.node_post_process(node, flatten_layer)
                        elif function_name == 'concat' or function_name == 'cat':
                            concat_layer = ConcatFunc(bb, node, self.node_map)
                            self.node_post_process(node, concat_layer)
                        elif function_name == 'softmax':
                            sigmoid_layer = SoftMaxLayer(bb, node, self.node_map)
                            self.node_post_process(node, sigmoid_layer)
                        elif function_name == 'sigmoid':
                            sigmoid_layer = SigmoidLayer(bb, node, self.node_map)
                            self.node_post_process(node, sigmoid_layer)
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
        self.relax_graph = bb.get()
    
    def fuse_op(self):
        '''
            TO DO: op fused PASS
        '''
        pass

    def gen_TensorIR(self):
        '''
            generate low level TensorIR from Relax IR
        '''
        from .tensorIR_layers.lower_to_tensorir_pass import LowerToTensorIRPass
        self.TensorIR = LowerToTensorIRPass()(self.relax_graph)


    def mlc_tune_op(self, mod_, op_name):
        logger.info("target: %s; compile_tir_target: %s" % (self.target, self.compile_tir_target))
        from tvm import meta_schedule as ms
        logger.info("op: "+ op_name)
        op_work_dir = self.work_dir + "op_%s" % (op_name)
        try:
            tuned_record = ms.tune_tir(mod_, target=self.target,
                                   work_dir=op_work_dir,
                                    task_name=self.task_name,
                                    max_trials_global=self.max_trials_global,
                                    num_trials_per_iter=self.num_trials_per_iter)
            tuned_sch = ms.tir_integration.compile_tir(tuned_record, mod_, target=self.compile_tir_target)
            new_func = tuned_sch.mod['main'].with_attr("global_symbol", op_name)
            return new_func
        except:
            raise ValueError("op name is not in Model, please check func::print_op(self)")
        
    def mlc_tune_tir(self, Model: IRModule, op_list=None,
                target=None, 
                work_dir="./tune_tmp/",
                task_name='main',
                max_trials_global=64,
                num_trials_per_iter=64,
                compile_tir_target='cuda'):
        if op_list is None:
            logger.info('tune all ops default')
            op_list = [x.name_hint for x in Model.functions]
            if 'main' in op_list:
                op_list.remove('main')
            print(len(op_list))

        if target is None and self.target is None:
            raise Warning("hardwear targer %s is None" % target)
        if target is not None:
            self.target = target

        self.work_dir=work_dir
        self.task_name=task_name
        self.max_trials_global=max_trials_global
        self.num_trials_per_iter=num_trials_per_iter
        self.compile_tir_target=compile_tir_target

        for i, op_name in enumerate(op_list):
            self.mlc_tune_op(Model, )
            mod_ = tvm.IRModule.from_expr(Model[op_name].with_attr("global_symbol", 'main'))
            new_func = self.mlc_tune_op(mod_, op_name)
            gv = Model.get_global_var(op_name)
            Model.update_func(gv, new_func)
            return Model
        self.tuned_TensorIR = Model
        return self.tuned_TensorIR
   
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
        if self.device == 'llvm':
            device = tvm.cpu()
        elif self.device == 'cuda':
            device = tvm.cuda()
        else:
            raise NotImplementedError("target device %s is not supported!" % self.device)
        self.get_Tensor_input()
        ex = relax.vm.build(self.TensorIR, self.device)
        vm = relax.vm.VirtualMachine(ex, device)
        self.TensorIR_output = []
        for i in self.tensor_input_list:
            self.TensorIR_output.append(vm['main'](i))


    def node_post_process(self, node, relax_layer):
        '''
            used to record graph node & supported to generate Relax IR
        '''
        self.node_map[node] = relax_layer.value