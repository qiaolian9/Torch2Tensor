from loguru import logger
from tvm import relax

from .base_layer import BaseLayer

class ChunkFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ChunkFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        chunks = self._source_node.args[1]
        dim = self._source_node.kwargs['dim']

        from ..register_relax.chunk import chunk
        out = self.bb.emit(chunk(x, chunks, dim), name_hint=self._name)
        logger.info("chunks_layer: " + self._name + " created")
        self.value = out