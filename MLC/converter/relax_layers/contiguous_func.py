# from loguru import logger
# import numpy as np

# from tvm import relax
# from .base_layer import BaseLayer
# from ..common_utils import (
#     get_shape,
#     map_reduce,
# )

# class ContiguousFunc(BaseLayer):
#     def __init__(self, source_node, module=None, auto_gen=True):
#         super(ContiguousFunc, self).__init__(source_node, module, auto_gen)

#     def get_flatten_attr(self):
#         attr_dict = dict(shape=None)
    
#         _output_shape = []
#         if "tensor_meta" in list(self._source_node.meta.keys()):
#             _output_shape.extend(
#                 map_reduce(self._source_node.meta["tensor_meta"], get_shape)
#             )
#         attr_dict["shape"] = _output_shape[0]

#         return attr_dict

#     def generate_node(self):
#         x = self.node_map[self._source_node.args[0]]
#         attr_dict = self.get_flatten_attr()
#         logger.debug(attr_dict['shape'])

#         out = self.bb.emit(relax.op.reshape(x, **attr_dict), name_hint=self._name)
#         logger.info("reshape_layer: " + self._name + " created")
#         self.value = out