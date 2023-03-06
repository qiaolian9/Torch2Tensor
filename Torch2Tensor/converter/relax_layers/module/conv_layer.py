import re
from loguru import logger
import torch.nn as nn
from tvm import relax
from ..base_layer import BaseLayer


class ConvLayer(BaseLayer):
    def __init__(self, bb, source_node, node_map=None, module=None, auto_gen=True):
        super(ConvLayer, self).__init__(bb, source_node, node_map, module, auto_gen)

    def get_conv_attr(self):
        conv_dim = int(re.findall(r"(?:Conv)([0-9]d*?)", str(self._module))[0])

        if isinstance(self._module, nn.Conv1d):  # con1d
            attr_dict = {
                "dilations": [1],  # list of ints defaults is 1
                "group": 1,  # int default is 1
                "pads": [0, 0],  # list of ints defaults to 0
                "strides": [1],  # list of ints  defaults is 1
            }

        else:
            attr_dict = dict(
                strides = (1, 1),
                padding = (0, 0),
                dilation = (1, 1),
                groups = 1,
                data_layout = "NCHW",
                kernel_layout = "OIHW",
                out_layout = None,
                out_dtype = None,
            )

        stride = self._module.stride
        padding = self._module.padding
        dilation = self._module.dilation
        groups = self._module.groups

        if isinstance(dilation, tuple):
            attr_dict["dilation"] = dilation
        else:
            attr_dict["dilation"] = [dilation]

        if isinstance(stride, tuple):
            attr_dict["strides"] = stride
        else:
            attr_dict["strides"] = [stride]

        if isinstance(padding, tuple):
            if len(padding) == 1:
                attr_dict["padding"] = padding * conv_dim * 2
            else:
                attr_dict["padding"] = padding * 2
        else:
            attr_dict["padding"] = [padding] * conv_dim * 2

        attr_dict["groups"] = groups

        return attr_dict

    def generate_node(self):
        kernel = self.create_params(self._name + "_weight", self._module.weight)
        if self._module.bias is not None:
            bias = self._module.bias
            shape = [bias.shape[0]]
            shape.extend([1] * len(self._module.weight.shape[2:]))
            bias = bias.reshape(shape)
            bias = self.create_params(self._name + "_bias", bias)
        x = self.node_map[self._source_node.args[0]]

        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        out = self.bb.emit(relax.op.nn.conv2d(x, kernel, **attr_dict), name_hint=self._name + ":conv")

        if self._name + "_bias" in self._init_tensor.keys():
            out = self.bb.emit(relax.op.add(out, bias), name_hint=self._name + ":bias")
        
        logger.info("conv_layer: " + self._name + " created")
        self.value = out
