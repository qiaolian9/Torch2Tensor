from loguru import logger
import torch.nn as nn
from tvm import relax
from ..base_layer import BaseLayer

class BatchNormLayer(BaseLayer):
    def __init__(self, bb: relax.BlockBuilder, source_node, node_map: dict, module=None, auto_gen=True):
        super(BatchNormLayer, self).__init__(bb, source_node, node_map, module, auto_gen)
    
    def get_batchnorm_attr(self):
        if isinstance(self._module, nn.BatchNorm2d):  # con1d
            attr_dict = dict(
                axis = 1,
                epsilon = self._module.eps,
                center = True,
                scale = True,
            )

        return attr_dict

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]] 
        gamma = self.create_params(self._name + '_weight', self._module.weight)
        beta = self.create_params(self._name + '_bias', self._module.bias)
        moving_mean = self.create_params(self._name + '_moving_mean', self._module.running_mean)
        moving_var = self.create_params(self._name + '_moving_var', self._module.running_var)

        attr_dict = self.get_batchnorm_attr()
        logger.debug(attr_dict)
        
        out = self.bb.emit(relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, **attr_dict), name_hint=self._name)

        logger.info("batch_layer: " + self._name + " created")
        self.value = out[0]

        