from loguru import logger

from tvm import relax
from ..base_layer import BaseLayer
from torch import fx

class FloorDivFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(FloorDivFunc, self).__init__(source_node, module, auto_gen)

    def get_value(self, index):
        x = self._source_node.args[index]
        if isinstance(x, fx.node.Node):
            return self.node_map[x]
        return x

    def generate_node(self):
        assert len(self._source_node.args) == 2
        x = self.get_value(0)
        y = self.get_value(1)
        if isinstance(x, relax.Var) or isinstance(y, relax.Var):
            if isinstance(x, relax.Expr) and isinstance(y, relax.Expr):
                pass
            elif isinstance(x, relax.Expr):
                y = relax.const(y, dtype=x.struct_info.dtype)
            elif isinstance(y, relax.Expr):
                x = relax.const(x, dtype=y.struct_info.dtype)
            else:
                raise Warning('floordiv error!')
            out = self.bb.emit(relax.op.floor_divide(x, y), name_hint=self._name)
        else:
            out = x // y
        
        
        logger.info("floordiv_layer: " + self._name + " created")
        self.value = out