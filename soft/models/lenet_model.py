import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from hyper_parameters import HParams, Layer

class Lenet(object):
    """
    Implementation of a slightly improved LeNet5 for MNIST
    """
    def __init__(self, hps, images, labels, mode, *args, **kwargs):
    
        self._images = images
        self.batch_size = hps.batch_size
        self.labels = labels
        self.mode = mode
        self.hps = hps
        self.global_step = tf.Variable(0, name='global_step', trainable=False)


    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
        if self.mode.startswith('train'):
            self._build_train_op()
        if self.mode.startswith('eval'):
            self._build_eval_op()
        self.summaries = tf.merge_all_summaries()

    def _build_model(self):
        """Build the core model within the graph.
           Implement convolution with kernel of different size in a single
           layer
    """

        x = tf.reshape(self._images, [-1, 28, 28, 1])
        x = tf.pad(x,[[0,0], [2,2], [2,2], [0,0]], "CONSTANT")

        print('Input image : ' + str(x))

        with tf.variable_scope('c1s2'):
            x = self._conv('conv1', x, 'DW0-5', 5, 1, 6, [1,1,1,1])
            print("Output C1 : " + str(x))
            x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'VALID')
            print("Output S2 : " + str(x))
            x = tf.tanh(x + self._fmap_bais(6))

        with tf.variable_scope('c3s4'):
            x = self._conv('conv2', x, 'DW1-5', 5, 6, 16, [1,1,1,1])
            print("Output C3 : " + str(x))
            x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'VALID')
            print("Output S4 : " + str(x))
            x = tf.tanh(x + self._fmap_bais(16))

        with tf.variable_scope('c5'):
            x = self._conv('conv3', x, 'DW2-5', 5, 16, 120, [1,1,1,1])
            print("Output C5 : " + str(x))
        
        with tf.variable_scope('f6'):
            x = self._fully_connected(self.batch_size, x, 84) 
            tf.nn.dropout(x, 0.5)
            print("Output f6 : " + str(x))

        with tf.variable_scope('output'):
            logits = self._fully_connected(self.batch_size, x, 10)
            self.predictions = tf.nn.softmax(logits)
        
        with tf.variable_scope('cost'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                    logits, self.labels)

            self.cost = tf.reduce_mean(xent, name='xent')


    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.scalar_summary('learning rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'adgrad':
            optimizer = tf.train.AdagradDAOptimizer(self.lrn_rate)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] 
        self.train_op = tf.group(*train_ops)
    

    def _build_eval_op(self):
        """ Compute the top1 and top5 accuracy from Imagenet"""
        predictions = self.predictions
        labels = tf.argmax(self.labels, 1)
        self.top1 = tf.cast(tf.nn.in_top_k(predictions, labels, 1), tf.int32)
        self.top5 = tf.cast(tf.nn.in_top_k(predictions, labels, 5), tf.int32)
        

    @staticmethod
    def _conv(name, x, kernel_name, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                kernel_name, 
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32, 
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            print('\t|CONV[%d x %d x %d x %d]|'%(filter_size, filter_size,
                in_filters, out_filters))
            return tf.nn.conv2d(x, kernel, strides, padding='VALID')

    @staticmethod
    def _fully_connected(batch_size, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [batch_size, -1])
        w = tf.get_variable(
            'DW', 
            [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', 
                [out_dim],
                initializer=tf.constant_initializer())
        print('\tFC[1 x %d]'%out_dim)
        return tf.nn.xw_plus_b(x, w, b)


    @staticmethod
    def _fmap_bais(nb_fmap):
        b = tf.get_variable('biases', 
                [1,1,1,nb_fmap],
                initializer=tf.constant_initializer())
        return b
