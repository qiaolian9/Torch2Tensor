import re
from loguru import logger
import torch.nn as nn
from tvm import relax
from .base_layer import BaseLayer


class LinearLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(LinearLayer, self).__init__(bb, source_node, node_map, module, auto_gen)

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        w = self.create_params(self._name + '_weight', self._module.weight)
        if self._module.bias is not None:
            bias = self.create_params(self._name + '_bias', self._module.bias)
        else:
            bias = None
            
        out = self.bb.emit(relax.op.linear(x, w, bias), name_hint=self._name)
        logger.info("linear_layer: " + self._name + " created")
        self.value = out
