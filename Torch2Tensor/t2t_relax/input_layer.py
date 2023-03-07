from loguru import logger
from tvm import relax
from tvm.script import relax as R
from .base_layer import BaseLayer


class InputLayer(BaseLayer):
    def __init__(self, bb, source_node, input_shape, dtype='float32', node_mape = None, module=None, auto_gen=True):
        super(InputLayer, self).__init__(bb, source_node, node_mape, module, auto_gen)
        self.input_shape = input_shape
        self.dtype = dtype
        self._generate_input()

    def _generate_input(self):
        logger.info("input_layer: " + self._name + " created")
        self.value = relax.Var(self._source_node.target, R.Tensor(self.input_shape, self.dtype))
        