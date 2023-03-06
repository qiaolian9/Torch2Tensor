from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer


class ConcatFunc(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(ConcatFunc, self).__init__(bb, source_node, node_map, module, auto_gen)

    def generate_node(self):
        x_ = []
        logger.info(self._source_node.args)
        for i in self._source_node.args[0]:
            x_.append(self.node_map[i])
        dim = None
        if len(self._source_node.args) == 2:
            dim = self._source_node.args[1]
        elif 'dim' in self._source_node.kwargs:
            dim = self._source_node.kwargs['dim']
        
        out = self.bb.emit(relax.op.concat(x_, dim), name_hint=self._name)
        
        logger.info("concat_layer: " + self._name + " created")
        self.value = out
