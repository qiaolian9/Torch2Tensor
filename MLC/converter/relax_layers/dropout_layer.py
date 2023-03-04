import re
from loguru import logger
import torch.nn as nn
from tvm import relax
from .base_layer import BaseLayer


class DropoutLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(DropoutLayer, self).__init__(bb, source_node, node_map, module, auto_gen)

    def get_conv_attr(self):

        if isinstance(self._module, nn.Dropout):  # con1d
            attr_dict = {
                'rate': 0.5
            }
            attr_dict["rate"] = self._module.p
        return attr_dict

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]]
        # dropout mode didn't work in eval mode
        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        out = self.bb.emit(relax.op.nn.dropout(x, **attr_dict), name_hint=self._name)
        logger.info("dropout_layer: " + self._name + " created")
        self.value = out[0]
