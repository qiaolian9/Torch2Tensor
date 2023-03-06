from loguru import logger
import numpy as np

from tvm import relax
from .base_layer import BaseLayer
from ..common_utils import (
    get_shape,
    map_reduce,
)

class TransposeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(TransposeFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        dim1 = self._source_node.args[1]
        dim2 = self._source_node.args[2]
        logger.info(dim1)
        logger.info(type(dim1))
        from ..register_relax.otherop.transpose import transpose
        out = self.bb.emit(transpose(x, dim1, dim2), name_hint=self._name)
        logger.info("transpose_layer: " + self._name + " created")
        self.value = out