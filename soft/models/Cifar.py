from cifar_input import build_input
from Dataset import Dataset

class Cifar(Dataset):

    def __init__(self, model_type, name, log_root=None, lrn_rate_thresholds=None,
            lrn_rate_values=None, stop=None):
        
        super(Cifar, self).__init__(name, model_type, log_root, lrn_rate_thresholds, lrn_rate_values,
                stop)

    def build_hps(self, data_path, hps, mode, *args):
        # Retrieve Cifar images
        self.images, self.labels = build_input(self.name, data_path, hps.batch_size, mode)
        # Definition of CNN parameters depending on the model
        if self.model_type.startswith('resnet20'):
            layers = [(3,16,3)] + [(3,16,16)]*6 + [(3,32,16)] + [(3,32,32)]*5 +[(3,64,32)] + [(3,64,64)]*5
            num_res_units = 3
            num_sub_units = [3]*3
        else:
            exit("Error: Model type " + self.model_type + " unknown for Cifar")
       
        if 'square' in self.model_type:
            m = 128
            layers = [(3,m,3)] + [(s,m,m)  for i,(s,n,d) in enumerate(layers) if i != 1] 
        
        # Specific case for size regularization
        if self.model_type.endswith('-5'):
            layers = [(5,n,d) for (s,n,d) in layers]
        elif self.model_type.endswith('-7'):
            layers = [(7,n,d) for (s,n,d) in layers]

        layers = [[l] for l in layers]
        
        hps.define_network_parameters(layers, num_res_units, num_sub_units)  

        hps.set_input_fmap(0,32)
