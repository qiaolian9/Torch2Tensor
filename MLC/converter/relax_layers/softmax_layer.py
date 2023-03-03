from loguru import logger
from tvm import relax
from .base_layer import BaseLayer


class SoftMaxLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(SoftMaxLayer, self).__init__(bb, source_node, node_map, module, auto_gen)

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        if self._module is not None:
            dim = self._module.dim
        else:
            dim = self._source_node.args[1]
        out = self.bb.emit(relax.op.nn.softmax(x, dim), name_hint=self._name)
        
        logger.info("softmax_layer: " + self._name + " created")
        self.value = out
