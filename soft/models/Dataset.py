from os import path, makedirs
from utility import sanitize_lrn_rates, read_lrn_rates
import numpy as np

class Dataset(object):
     
    
    def __init__(self, name, model_type, log_root, lrn_rate_thresholds, lrn_rate_values,
            stop):
        self.name = name
        self.model_type = model_type
        self.lrn_rate_thresholds = lrn_rate_thresholds
        self.lrn_rate_values = lrn_rate_values
        self.stop_step = stop

        # Setup environment folder for learning rate storage and hand
        # modifications
        if log_root is not None :
            self.lrn_rate_file = log_root + "/lrn_rate.txt"  
            if not path.exists(log_root):
                makedirs(log_root)
         
        if (lrn_rate_thresholds is not None and
                lrn_rate_values is  not None):
        
            with open(self.lrn_rate_file,'w+') as file:
                file.write('\t'.join(str(thr) for thr in
                    self.lrn_rate_thresholds) + "\n")
                file.write('\t'.join(str(thr) for thr in self.lrn_rate_values))

            self.last_edited = path.getmtime(self.lrn_rate_file)
   

    def learning_rate(self, step):
        """ Define the learning rate depending on the iteration step"""
        if(path.getmtime(self.lrn_rate_file) != self.last_edited):
            self.lrn_rate_thresholds, self.lrn_rate_values, self.stop_step =read_lrn_rates(self.lrn_rate_file, self.name)
            self.last_edited = path.getmtime(self.lrn_rate_file)

        valid_threshes = np.array(self.lrn_rate_thresholds) <= step
        true_indexes   = [ i for i, x in enumerate(valid_threshes) if x]
        return self.lrn_rate_values[true_indexes[-1]]

    def stop(self, step):
        """ Maximum number of iterations definition"""
        return step > self.stop_step

