import re
from loguru import logger
import torch.nn as nn
from tvm import relax
from ..base_layer import BaseLayer


class ConvLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(ConvLayer, self).__init__(bb, source_node, node_map, module, auto_gen)

    def get_conv_attr(self):
        if isinstance(self._module, nn.Conv2d):  # con1d
            attr_dict = dict(
                strides = self._module.stride,
                padding = self._module.padding,
                dilation = self._module.dilation,
                groups = self._module.groups,
                data_layout = "NCHW",
                kernel_layout = "OIHW",
                out_layout = None,
                out_dtype = None,
            )

        return attr_dict

    def generate_node(self):
        kernel = self.create_params(self._name + "_weight", self._module.weight)
        x = self.node_map[self._source_node.args[0]]

        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        out = self.bb.emit(relax.op.nn.conv2d(x, kernel, **attr_dict), name_hint=self._name + ":conv")


        if self._module.bias is not None:
            bias = self.create_params(self._name + "_bias", self._module.bias)
            bias = relax.op.reshape(bias, (1, -1, 1, 1))
            out = self.bb.emit(relax.op.add(out, bias), name_hint=self._name + ":bias")
        
        logger.info("conv_layer: " + self._name + " created")
        self.value = out
