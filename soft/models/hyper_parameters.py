import utility as ut
class HParams(object):
    def __init__(self, batch_size, num_classes, lrn_rate, res_mod, optimizer,
            dataset, *args, **kwargs):
        super(HParams, self).__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lrn_rate = lrn_rate
        self.res_mod = res_mod
        self.optimizer = optimizer
        self.dataset = dataset
        
    def define_network_parameters(self, layers, num_res_units, num_sub_units):
        self.layers = [Layer(l, i) for i,l in enumerate(layers)]
        self.num_res_units = num_res_units
        self.num_sub_units = num_sub_units

    def get_layer(self, *args, **kwargs):
        if len(args) == 1:
            return self.layers[args[0]].filters[0]
        elif len(args) == 2:
            return self.layers[args[0]].get_filters(args[1])
            
    def set_input_fmap(self, layer_nb, fmap):
        self.layers[layer_nb].fmap = float(fmap)

    def get_input_size(self, layer_nb):
        return self.layers[layer_nb].fmap


class Layer(object):
    def __init__(self, layer_tuple,  layer_nb, *args,
            **kwargs):
        super(Layer, self).__init__(*args, **kwargs)
        self.layer_nb = layer_nb
        self.filters = [FilterType(s, n, d) for (s, n, d) in layer_tuple]
        self.tot_num_filters = sum([ f.num_filters for f in self.filters])
        self.depth = self.filters[0].depth

    def get_filters(self, filter_size):
        return next((x for x in self.filters if x.filter_size == filter_size), None)

    def get_max_size(self):
        return max([ ftype.filter_size for ftype in self.filters])


class FilterType(object):
    def __init__(self, filter_size, num_filters, depth, *args,
            **kwargs):
        super(FilterType, self).__init__(*args, **kwargs)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.depth = depth

