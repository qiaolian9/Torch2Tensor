from loguru import logger

from tvm import relax
from ..base_layer import BaseLayer

class MulFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MulFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        assert len(self._source_node.args) == 2
        x = self.node_map[self._source_node.args[0]]
        y = self.node_map[self._source_node.args[1]]

        out = self.bb.emit(relax.op.multiply(x, y), name_hint=self._name)
        
        logger.info("mul_layer: " + self._name + " created")
        self.value = out