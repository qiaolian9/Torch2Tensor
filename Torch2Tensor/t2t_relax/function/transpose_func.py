from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer

class TransposeFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(TransposeFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        dim1 = self._source_node.args[1]
        dim2 = self._source_node.args[2]
        full_idx = list(range(len(x.struct_info.shape)))
        full_idx[dim1], full_idx[dim2] = dim2, dim1
        out = self.bb.emit(relax.op.permute_dims(x, full_idx), name_hint=self._name)
        logger.info("transpose_layer: " + self._name + " created")
        self.value = out