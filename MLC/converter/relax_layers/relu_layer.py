from loguru import logger
from tvm import relax
from .base_layer import BaseLayer
import torch.nn as nn


class ReluLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(ReluLayer, self).__init__(bb, source_node, node_map, module, auto_gen)


    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        if isinstance(self._module, nn.ReLU):  
            out = self.bb.emit(relax.op.nn.relu(x), name_hint=self._name)
        elif isinstance(self._module, nn.ReLU6):
            from ..register_relax.relu6 import relu6
            out = self.bb.emit(relu6(x), name_hint=self._name)
        elif isinstance(self._module, nn.SiLU):
            out = self.bb.emit(relax.op.nn.silu(x), name_hint=self._name)
        logger.info("relu_layer: " + self._name + " created")
        self.value = out