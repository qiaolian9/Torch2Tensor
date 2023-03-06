from loguru import logger
from tvm import relax
from ..base_layer import BaseLayer
from ...common_utils import get_function_name

class Pool2dFunc(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(Pool2dFunc, self).__init__(bb, source_node, node_map, module, auto_gen)

    def get_avgpool2d_attr(self):
        func_name = get_function_name(self._source_node.target)
        if func_name.startswith('adaptive'):
            attr_dict = dict(
                output_size = None,
                layout = "NCHW",
                out_layout = None,
            )

            attr_dict['output_size'] = self.get_value_by_key_or_index("output_size", 1, None)
            self.pooltype = 'adaptiveavgpool2d'
        else:
            attr_dict = dict(
                pool_size = (1, 1),
                strides = (1, 1),
                padding = (0, 0),
                dilation = (1, 1),
                ceil_mode = False,
                layout = "NCHW",
                out_layout = "NCHW",
            )
            attr_dict['pool_size'] = self.get_value_by_key_or_index("kernel_size", -1, 1)
            attr_dict['strides'] = self.get_value_by_key_or_index("stride", -1, 1)
            attr_dict['padding'] = self.get_value_by_key_or_index("padding", -1, 0)
            attr_dict['dilation'] = self.get_value_by_key_or_index('dilation', -1, 1)
            attr_dict['ceil_mode'] = self.get_value_by_key_or_index('ceil', -1, False)
            if func_name == 'avg_pool2d':
                self.pooltype = 'avgpool2d'
            elif func_name == 'boolean_dispatch':
                # F.maxpool
                self.pooltype = 'maxpool2d'
            else:
                raise NotImplementedError('op %s is not implemented now!' % self._name)
        return attr_dict

    def generate_node(self):
        print(self._name)
        x = self.node_map[self._source_node.args[0]]

        attr_dict = self.get_avgpool2d_attr()
        logger.debug(attr_dict)
        if self.pooltype == 'avgpool2d':
            from ...register_relax.nn.pool.avgpool import avg_pool2d
            out = self.bb.emit(avg_pool2d(x, **attr_dict), name_hint=self._name)
        elif self.pooltype == 'maxpool2d':
            attr_dict['pool_size'] = self._source_node.args[1]
            out = self.bb.emit(relax.op.nn.max_pool2d(x, **attr_dict), name_hint=self._name)
        elif self.pooltype == 'adaptiveavgpool2d':
            out = self.bb.emit(relax.op.nn.adaptive_avg_pool2d(x, **attr_dict), name_hint=self._name)
        else:
            raise NotImplementedError('op type %s is not implemented now!' % self._name)
        logger.info("pool2d_layer: " + self._name + " created")
        self.value = out