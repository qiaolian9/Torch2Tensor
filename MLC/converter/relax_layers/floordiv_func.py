from loguru import logger

from tvm import relax
from .base_layer import BaseLayer

class FloorDivFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(FloorDivFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        assert len(self._source_node.args) == 2
        x = self.node_map[self._source_node.args[0]]
        y = self._source_node.args[1]
        logger.debug(self._source_node.kwargs)
        logger.debug(self._source_node.args)
        logger.debug(x)
        logger.debug(type(x))
        if isinstance(y, int):
            y = relax.const(y)
        else:
            y = self.node_map[y]

        out = self.bb.emit(relax.op.floor_divide(x, y), name_hint=self._name)
        
        logger.info("floordiv_layer: " + self._name + " created")
        self.value = out