from loguru import logger

from .base_layer import BaseLayer

class OutputLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(OutputLayer, self).__init__(bb, source_node, node_map, module, auto_gen)
        self.generate_output()


    def generate_output(self):
        output = self.node_map[self._source_node.args[0]]
        logger.info("output_layer: " + self._name + " created")
        self.value = self.bb.emit_output(output)
        