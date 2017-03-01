import imagenet_input
from Dataset import Dataset


class Imagenet(Dataset):

    def __init__(self, model_type, log_root=None, lrn_rate_thresholds=None,
            lrn_rate_values=None, stop=None):
        super(Imagenet, self).__init__('imagenet', model_type, log_root, lrn_rate_thresholds, lrn_rate_values,
                stop)


    def build_hps(self, data_path, hps, mode):
        if mode == 'train':
            self.images, self.labels = imagenet_input.distorded_input(batch_size=hps.batch_size,
                                                                 data_path=data_path)
        else:
            self.images, self.labels = imagenet_input.inputs(hps.batch_size, data_path=data_path)
        
        # Definition of CNN parameters depending on the model
        if self.model_type.startswith('resnet8'):
            layers = [(7,64,3)] + \
                     [(3,64,64)]*2 + \
                     [(3,128,64)] + [(3,128,128)] + \
                     [(3,256,128)] + [(3,256,256)]
            num_res_units = 3
            num_sub_units = [1]*4
        elif self.model_type.startswith('resnet18'):
            layers = [(7,64,3)] + \
                     [(3,64,64)]*4 + \
                     [(3,128,64)] + [(3,128,128)]*3 + \
                     [(3,256,128)] + [(3,256,256)]*3 + \
                     [(3,512,256)] +[(3,512,512)]*3
            num_res_units = 4
            num_sub_units = [2]*4
       
        else:
            exit("Error: Model type " + self.model_type + " unknown for Imagenet")

        # Specific case for size regularization
        if self.model_type.endswith('-5'):
            layers = [(5,n,d) for (s,n,d) in layers]
        elif self.model_type.endswith('-7'):
            layers = [(7,n,d) for (s,n,d) in layers]

        if 'square' in self.model_type:
            layers = [(s,512,512) if i != 1 else (s,512,3) for i,(s,n,d) in
                    enuerate(layers)] 

        hps.define_network_parameters(layers, num_res_units, num_sub_units)  
