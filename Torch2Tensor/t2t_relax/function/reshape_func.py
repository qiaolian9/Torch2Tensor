from loguru import logger

from tvm import relax
from ..base_layer import BaseLayer
from ...utils import (
    get_shape,
    map_reduce,
)

class ReshapeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ReshapeFunc, self).__init__(source_node, module, auto_gen)

    def get_flatten_attr(self):
        attr_dict = dict(shape=None)
    
        _output_shape = []
        if "tensor_meta" in list(self._source_node.meta.keys()):
            _output_shape.extend(
                map_reduce(self._source_node.meta["tensor_meta"], get_shape)
            )
        attr_dict["shape"] = _output_shape[0]

        return attr_dict

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        attr_dict = self.get_flatten_attr()
        logger.debug(attr_dict['shape'])

        # shape = self._source_node.args[1]
        # out = self.bb.emit(relax.op.reshape(x, shape), name_hint=self._name)
        out = self.bb.emit(relax.op.reshape(x, **attr_dict), name_hint=self._name)
        logger.info("reshape_layer: " + self._name + " created")
        self.value = out