from loguru import logger

from ..base_layer import BaseLayer

class StochasticDepthFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(StochasticDepthFunc, self).__init__(source_node, module, auto_gen)


    def generate_node(self):
        out = self.node_map[self._source_node.args[0]]
        
        logger.info("stochasticdepth_layer: " + self._name + " created")
        self.value = out