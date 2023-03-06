from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer

class AddFunc(BaseLayer):
    def __init__(self, bb, source_node, node_map, module=None, auto_gen=True):
        super().__init__(bb, source_node, node_map, module, auto_gen)

    def generate_node(self):
        assert len(self._source_node.args) == 2
        x = self.node_map[self._source_node.args[0]]
        y = self.node_map[self._source_node.args[1]]

        out = self.bb.emit(relax.op.add(x, y), name_hint = self._name)
        
        logger.info("add_layer: " + self._name + " created")
        self.value = out