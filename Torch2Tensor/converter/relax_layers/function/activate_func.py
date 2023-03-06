from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer
from ...common_utils import get_function_name

class ActivateFunc(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(ActivateFunc, self).__init__(bb, source_node, node_map, module, auto_gen)


    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]

        function_name = get_function_name(self._source_node.target)
        if function_name == 'relu':
            out = self.bb.emit(relax.op.nn.relu(x), name_hint=self._name)
        elif function_name == 'softmax':
            dim = self._source_node.args[1]
            out = self.bb.emit(relax.op.nn.softmax(x, dim), name_hint=self._name)
        elif function_name == 'sigmoid':
            out = self.bb.emit(relax.op.sigmoid(x), name_hint=self._name)
        else:
            raise Warning("func %s is not implemented now!" % self._name)

        logger.info("activate_layer: " + self._name + " created")
        self.value = out