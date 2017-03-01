import tensorflow as tf
from resnet_model import ResNet

class ResNetSingleGPU(ResNet):



class ResNetMultiGPUS():

    def __init__(self, *args, **kwargs):
        self.nb_gpus = 8


    def build_graph(self):
        """ Build model and training operations according to a multiple GPUS
        architecture """
        if self.mode.startswith('train'):
            self._build_train_op()
        if self.mode.startswith('eval'):
            self._build_model()
            self._build_eval_op()
        self.summaries = tf.merge_all_summaries()
          

    def _build_train_op(self):
        """ Build a training operation according to a
        multi-gpus_architecture"""

        opt = self._build_optimizer()
        tower_grads = []
        for i in range(self.nb_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower',i)) as scope:
                    # Build the inference
                    self._build_model()
                    
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()        
                    
                    # Compute the gradient and store them in the tower
                    tower_grads.append(opt.compute_gradients(self.cost))
     
        # Compute the mean of each gradient taken from the different GPUs
        grads = average_gradients(tower_grads)
        apply_op = opt.apply_gradients(grads, global_step=global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)


    def _build_optimizer(self):
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.scalar_summary('learning rate', self.lrn_rate)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate)

        return optimizer


    def average_gradients(tower_grads):
      """Calculate the average gradient for each shared variable across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat_v2(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
      return average_grads

