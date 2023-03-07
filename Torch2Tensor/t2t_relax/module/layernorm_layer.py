from loguru import logger
import torch.nn as nn
from tvm import relax
from ..base_layer import BaseLayer

class LayerNormLayer(BaseLayer):
    def __init__(self, bb: relax.BlockBuilder, source_node, node_map: dict, module=None, auto_gen=True):
        super(LayerNormLayer, self).__init__(bb, source_node, node_map, module, auto_gen)
    
    def get_layernorm_attr(self):
        if isinstance(self._module, nn.LayerNorm):  
            attr_dict = dict(
                axes=-1,
                epsilon = self._module.eps,
                center = True,
                scale = True,
            )

        return attr_dict

    def generate_node(self):
        x = self.node_map[self._source_node.args[0]] 
        gamma = self.create_params(self._name + '_weight', self._module.weight)
        beta = self.create_params(self._name + '_bias', self._module.bias)

        attr_dict = self.get_layernorm_attr()
        logger.debug(attr_dict)
        
        out = self.bb.emit(relax.op.nn.layer_norm(x, gamma, beta, **attr_dict), name_hint=self._name)
        logger.info("layernorm_layer: " + self._name + " created")
        self.value = out