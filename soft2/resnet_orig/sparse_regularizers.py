import tensorflow as tf

class Regularizer(object):
    #def __init__(self, *args, **kwargs):
    #    super(Regularizer, self).__init__(*args, **kwargs)
    def _build_reg(self): pass
    def _build_model(self, *args, **kwargs):
        super(Regularizer, self)._build_model(*args, **kwargs)
        self.cost += self._build_reg();

class LayerDepth(Regularizer):
    def _build_reg(self):
        """ L2,1 loss on the number of kernels. Should enforce their sparsity"""
        
        kernel_l2_norms=[]
        for var in tf.trainable_variables():
            if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
                layer_kernel_l2_norms = tf.sqrt(tf.reduce_sum(tf.square(var), [0,1,2])) # [ |k_{i1}|_2, ..., |k_{id_i}|_2 ]
                tf.histogram_summary('Kernel l2 norms - '+var.op.name, layer_kernel_l2_norms) ### CB2:SPEED
                kernel_l2_norms.append(layer_kernel_l2_norms)
        
        reg_val = tf.reduce_sum(tf.concat(0, layer_kernel_l2_norms))
        
        return reg_val

class Kernel_Size(Regularizer):
    pass

class Complexity(Regularizer):
    pass
