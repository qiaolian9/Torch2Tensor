from ..relax_utils import map_params


class BaseLayer(object):
    def __init__(self, bb, source_node, node_map, module=None, auto_gen=True):
        self.bb = bb
        self._source_node = source_node
        self._module = module
        self.node_map = node_map
        self._auto_gen = auto_gen

        self._init_tensor = dict()
        self._name = self._source_node.name
        if self._auto_gen:
            self.generate_node()

    def create_params(self, param_name, param):
        if param_name not in self._init_tensor.keys():
            param_tensor = map_params(param)
            self._init_tensor[param_name] = param_tensor
            return param_tensor
        else:
            raise KeyError("param name  %s is already used!" % (param_name)) 


    def generate_node(self):
        pass

    def get_value_by_key_or_index(self, key, index, default=None):
        if key in self._source_node.kwargs:
            return self._source_node.kwargs[key]
        elif index < len(self._source_node.args):
            return self._source_node.args[index]
        else:
            return default


