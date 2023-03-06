from loguru import logger

from tvm import relax
from ..base_layer import BaseLayer

class MeanFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MeanFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        
        x = self.node_map[self._source_node.args[0]]
        axis = None 
        if len(self._source_node.args) == 2:
            axis = self._source_node.args[1]

        out = self.bb.emit(relax.op.mean(x, axis), name_hint=self._name)
        
        logger.info("mean_layer: " + self._name + " created")
        self.value = out