import re
from loguru import logger
import tvm
from tvm import relax
from .base_layer import BaseLayer
from ..common_utils import get_function_name

class AvgPool2dFunc(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(AvgPool2dFunc, self).__init__(bb, source_node, node_map, module, auto_gen)

    def get_avgpool2d_attr(self):
        func_name = get_function_name(self._source_node.target)
        if func_name == 'avg_pool2d':
            attr_dict = dict(
                pool_size = (1, 1),
                strides = (1, 1),
                padding = (0, 0),
                dilation = (1, 1),
                ceil_mode = False,
                layout = "NCHW",
                out_layout = None,
            )
            print(self._source_node)
            attr_dict['pool_size'] = self.get_value_by_key_or_index("kernel_size", 1, None)
            attr_dict['strides'] = self.get_value_by_key_or_index("stride", 2, attr_dict['pool_size'])
            attr_dict['padding'] = self.get_value_by_key_or_index("padding", 3, 0)
            attr_dict['dilation'] = self.get_value_by_key_or_index('dilation', 4)
            attr_dict['ceil_mode'] = self.get_value_by_key_or_index('ceil', 5, False)
            self.pooltype = 'avgpool2d'
        else:
            raise NotImplementedError('op %s is not implemented now!' % self._name)
        return attr_dict

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]

        attr_dict = self.get_avgpool2d_attr()
        logger.debug(attr_dict)
        if self.pooltype == 'avgpool2d':
            from ..register_relax.avgpool import avg_pool2d
            out = avg_pool2d(x, **attr_dict)
            
            out = self.bb.emit(out, name_hint=self._name)
        else:
            raise NotImplementedError('op type %s is not implemented now!' % self._name)
        logger.info("pool2d_layer: " + self._name + " created")
        self.value = out