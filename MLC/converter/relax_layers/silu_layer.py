from loguru import logger
from tvm import relax
from .base_layer import BaseLayer


class SiLULayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(SiLULayer, self).__init__(bb, source_node, node_map, module, auto_gen)


    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        out = self.bb.emit(relax.op.nn.silu(x), name_hint=self._name)
        
        logger.info("silu_layer: " + self._name + " created")
        self.value = out
