from loguru import logger
import torch.nn as nn
from tvm import relax
from ..base_layer import BaseLayer


class Pool2dLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(Pool2dLayer, self).__init__(bb, source_node, node_map, module, auto_gen)

    def get_conv_attr(self):
        if isinstance(self._module, (nn.MaxPool2d, nn.AvgPool2d)): 
            attr_dict = dict(
                pool_size = (1, 1),
                strides = (1, 1),
                padding = (0, 0),
                dilation = (1, 1),
                ceil_mode = False,
                layout = "NCHW",
                out_layout = "NCHW",
            )
            attr_dict['pool_size'] = self._module.kernel_size
            attr_dict['strides'] = self._module.stride
            attr_dict['padding'] = self._module.padding
            attr_dict['dilation'] = self._module.dilation if hasattr(self._module, 'dilation') else 1
            attr_dict['ceil_mode'] = self._module.ceil_mode
            self.pooltype = 'maxpool2d' if isinstance(self._module, (nn.MaxPool2d)) else 'avgpool2d'
        elif isinstance(self._module, nn.AdaptiveAvgPool2d):
            attr_dict = dict(
                output_size = None,
                layout = "NCHW",
                out_layout = None,
            )
            attr_dict['output_size'] = self._module.output_size
            self.pooltype = 'adaptiveavgpool2d'
        else:
            raise NotImplementedError('op %s is not implemented now!' % self._name)
        return attr_dict

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]

        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        if self.pooltype == 'maxpool2d':
            out = self.bb.emit(relax.op.nn.max_pool2d(x, **attr_dict), name_hint=self._name)
        elif self.pooltype == 'avgpool2d':
            from ...register_relax.nn.pool.avgpool import avg_pool2d
            out = self.bb.emit(avg_pool2d(x, **attr_dict), name_hint=self._name)
        elif self.pooltype == 'adaptiveavgpool2d':
            out = self.bb.emit(relax.op.nn.adaptive_avg_pool2d(x, **attr_dict), name_hint=self._name)
        else:
            raise NotImplementedError('op type %s is not implemented now!' % self._name)
        logger.info("pool2d_layer: " + self._name + " created")
        self.value = out
