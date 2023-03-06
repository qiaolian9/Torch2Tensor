from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer
import torch.nn as nn


class ActivateLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(ActivateLayer, self).__init__(bb, source_node, node_map, module, auto_gen)


    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        if isinstance(self._module, nn.ReLU):  
            out = self.bb.emit(relax.op.nn.relu(x), name_hint=self._name)
        elif isinstance(self._module, nn.ReLU6):
            from ...register_relax.nn.activate.relu6 import relu6
            out = self.bb.emit(relu6(x), name_hint=self._name)
        elif isinstance(self._module, nn.SiLU):
            out = self.bb.emit(relax.op.nn.silu(x), name_hint=self._name)
        elif isinstance(self._module, nn.Hardswish):
            from ...register_relax.nn.activate.hardswish import hardswish
            out = self.bb.emit(hardswish(x), name_hint=self._name)

        elif isinstance(self._module, nn.Sigmoid):
            out = self.bb.emit(relax.op.sigmoid(x), name_hint=self._name)
        elif isinstance(self._module, nn.Hardsigmoid):
            from ...register_relax.nn.activate.hardsigmoid import hardsigmoid
            out = self.bb.emit(hardsigmoid(x), name_hint=self._name)
        elif isinstance(self._module, nn.Softmax):
            if self._module is not None:
                dim = self._module.dim
            else:
                dim = self._source_node.args[1]
            out = self.bb.emit(relax.op.nn.softmax(x, dim), name_hint=self._name)

        logger.info("activate_layer: " + self._name + " created")
        self.value = out