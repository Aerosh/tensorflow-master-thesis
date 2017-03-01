from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import json
from Dataset import Dataset

MNIST_PIXELS = 28*28

class Mnist(Dataset):

    def __init__(self,model_type, log_root=None, lrn_rate_thresholds=None,
            lrn_rate_values=None, stop=None):
        super(Mnist, self).__init__('mnist', model_type, log_root, lrn_rate_thresholds, lrn_rate_values,
                stop)

    def next_batch(self):
        if self.mode.endswith('train'):
            vector, labels = self.input_pipeline.train.next_batch(self.batch_size)

        else:
            vector, labels = self.input_pipeline.test.next_batch(self.batch_size)

        return vector, labels

    def build_hps(self, data_path, hps, mode, conf):
        """ Build hyperparameters from file or manually
        """
        # Retrieve MNIST images
        self.input_pipeline = input_data.read_data_sets(data_path, one_hot=True)
        self.batch_size = hps.batch_size
        self.mode = mode

        # Definition of CNN parameters depending on the model

        # Manual definition of the network from file
        if conf == '':
            if self.model_type.startswith('resnet4'):
                layers = [(5,32,1)] + [(5,32,32)]*2 
                num_res_units = 1
                num_sub_units = [1] 

            elif self.model_type.startswith('resnet10'):
                layers = [(5,16,1)] + [(5,32,16)] + [(5,32,32)]*3 + [(5,64,32)] + [(5,64,64)]*3
                num_res_units = 2
                num_sub_units = [2,2]

            elif self.model_type.startswith('lenet'):
                layers = [(5, 6, 1)] + [(5,16,6)] + [(5,120,16)] 
                num_res_units = 0
                num_sub_units = [0]
            else:
                exit("Error: Model type " + self.model_type + " unknown for Cifar")

            if 'square' in self.model_type:
                layers =  [(5,64,1)] + [(s,64,64) for i, (s,n,d) in
                        enumerate(layers) if i!=1 ]  

            # Specific case for size regularization
            if '-' in self.model_type :
                fs = int(self.model_type.split('-')[-1])
                layers = [(fs,n,d) for (s,n,d) in layers]
     
            layers = [[l] for l in layers]
        # From config file
        else :
            datafile = open(conf,'r')
            data = json.load(datafile)
            layers = data['arch']
            num_res_units = data['num_res_units']
            num_sub_units = data['num_sub_units']


        images_placeholder = tf.placeholder(tf.float32, shape=(hps.batch_size, MNIST_PIXELS))
        labels_placeholder = tf.placeholder(tf.float32, shape=(hps.batch_size, 10))
        
        self.images, self.labels = images_placeholder, labels_placeholder
        hps.define_network_parameters(layers, num_res_units, num_sub_units)  
        
        # Add image size as the first input feature map
        hps.set_input_fmap(0, 28)
