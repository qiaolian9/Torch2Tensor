from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer

class GetItemFunc(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(GetItemFunc, self).__init__(bb, source_node, node_map, module, auto_gen)

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        if isinstance(x, (list, tuple, relax.ShapeExpr, relax.Tuple)):
            out = x[self._source_node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.struct_info, relax.TupleStructInfo):
                out = self.bb.emit(relax.TupleGetItem(x, self._source_node.args[1]))
            else:
                assert isinstance(x.struct_info, relax.TensorStructInfo)
                begin = []
                end = []
                stride = []
                axes = []
                expand_dim = []
                i = 0
                shape = x.struct_info.shape
                for index in self._source_node.args[1]:
                    if isinstance(index, int):
                        begin.append(index)
                        end.append(index + 1)
                        stride.append(1)
                        axes.append(i)
                        i = i + 1
                    elif isinstance(index, slice):
                        begin.append(0 if index.start is None else index.start)
                        end.append(shape[i] if index.stop is None else index.stop)
                        stride.append(1 if index.step is None else index.step)
                        axes.append(i)
                        i = i + 1
                    elif index is None:
                        expand_dim.append(i)
                        i = i + 1
                    else:
                        raise ValueError("Unsupported index type: " + str(type(index)))
                while i < len(shape):
                    begin.append(0)
                    end.append(shape[i])
                    axes.append(i)
                    i = i + 1
                sliced = self.bb.emit(relax.op.strided_slice(x, axes, begin, end, stride))
                sliced_shape = list(self.shape_of(sliced))
                for i in expand_dim:
                    sliced_shape.insert(i, 1)
                out = self.bb.emit(relax.op.reshape(sliced, sliced_shape))
        else:
            assert False
        logger.info("getitem_layer: " + self._name + " created")
        self.value = out
