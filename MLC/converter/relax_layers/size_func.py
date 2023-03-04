from loguru import logger

from tvm import relax
from .base_layer import BaseLayer

class SizeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(SizeFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        assert len(self._source_node.args) == 1
        x = self.node_map[self._source_node.args[0]]

        out = self.bb.emit(relax.op.shape_of(x), name_hint=self._name)
        
        logger.info("size_layer: " + self._name + " created")
        self.value = out