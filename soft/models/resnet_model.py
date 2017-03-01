"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from hyper_parameters import HParams, Layer

class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode, *args, **kwargs):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Batch normalization operations array
        self._extra_train_ops = []

        # Layer number tracking for naming variable
        self.layer = 0

        # Input feature map
        self.fmap = 0 
        self.verbose = ''


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

        activate_before_residual = [True, False, False, False]
        stride_array = [1, 2, 2, 2]
        init_stride = stride_array[0]
        if self.hps.dataset == "imagenet":
            init_stride = stride_array[1]

        if self.hps.res_mod == 'plain':
            res_func = self._plain
        elif self.hps.res_mod == 'residual':
            res_func = self._residual
        else:
            res_func = self._bottleneck_residual

        # Set up input feature map from the hyper parameters object
        self.fmap = self.hps.get_input_size(self.layer)
        # Initial non-residual unit
        with tf.variable_scope('init'):
            if self.hps.dataset == "mnist":
                x = tf.reshape(self._images, [-1, 28, 28, 1])
            else:
                x = self._images

            layer = self.hps.layers[self.layer]
            output_fmap = []
            self.verbose = '\t'
            # Multiple kernel size along the same layer
            for filter_type in layer.filters :
                output_fmap.append(self._conv('init_conv', x, 'DW' + str(self.layer) + '-' +
                        str(filter_type.filter_size),
                        filter_type.filter_size, layer.depth, filter_type.num_filters,
                        self._stride_arr(init_stride)))
            # Merge all the produced output feature maps
            x = tf.concat(3, output_fmap)
            print(self.verbose)
 
            self.layer += 1

            if self.hps.dataset == 'imagenet':
                x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                self.fmap /= 2.0    

        #  Build graph residual units
        for i in range(self.hps.num_res_units):
            x = self._build_unit(res_func, x, self._stride_arr(stride_array[i]),
                activate_before_residual[i], 'DW', i+1, self.hps.num_sub_units[i])

            self.fmap /= 2.0

        with tf.variable_scope('unit_last'):
            print('unit_last')
            x = self._batch_norm('final_bn', x)
            x = tf.nn.relu(x, name='relu')
            x = self._global_avg_pool(x)
            print('\t[RELU]\n\t[AVERAGE POOL]')

        with tf.variable_scope('logit'):
            print('logit')
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            if self.hps.dataset == "mnist":
                labels = tf.cast(self.labels, tf.int32)
            else:
                labels = self.labels

            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits , labels)
            self.cost = tf.reduce_mean(xent, name='xent')


        


    def _build_unit(self, res_func, input_fmap, stride_arr, activate_before_residual, kernel_name, unit_nb, num_sub_units):

        with tf.variable_scope('unit_%d_0' % unit_nb):
            print('unit_%d_0' % unit_nb)
            output_fmap = res_func(input_fmap, stride_arr,activate_before_residual, kernel_name)
        for i in range(1, num_sub_units):
            with tf.variable_scope('unit_%d_%d' % (unit_nb, i)):
                output_fmap = res_func(output_fmap, self._stride_arr(1) , False, kernel_name)
        return output_fmap


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

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
    

    def _build_eval_op(self):
        """ Compute the top1 and top5 accuracy from Imagenet"""
        predictions = self.predictions
        labels = tf.argmax(self.labels, 1)
        self.top1 = tf.cast(tf.nn.in_top_k(predictions, labels, 1), tf.int32)
        self.top5 = tf.cast(tf.nn.in_top_k(predictions, labels, 5), tf.int32)


    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.histogram_summary(mean.op.name, mean)
                tf.histogram_summary(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _residual(self, x, stride, activate_before_residual=False, kernel_name='DW'):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = tf.nn.relu(x, name='relu')
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('sub1'):
            layer = self.hps.layers[self.layer]

            # Input and output feature map definition
            self.hps.set_input_fmap(self.layer, self.fmap)
            output_fmap = []

            # Display init
            self.verbose = '\t'

            for filter_type in layer.filters :
                output_fmap.append( 
                        self._conv('conv1', x, kernel_name + str(self.layer) + '-' + str(filter_type.filter_size) ,
                            filter_type.filter_size,
                            filter_type.depth,
                            filter_type.num_filters,
                            stride))
            x = tf.concat(3, output_fmap)
            print(self.verbose)
            self.layer += 1
            

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = tf.nn.relu(x, name='relu')
            layer = self.hps.layers[self.layer]

            # Input and output feature map definition
            self.hps.set_input_fmap(self.layer, self.fmap)
            output_fmap = []

            # Display init
            self.verbose = '\t'

            for filter_type in layer.filters :
                output_fmap.append( 
                        self._conv('conv2', x, kernel_name + str(self.layer) + '-' + str(filter_type.filter_size) ,
                            filter_type.filter_size,
                            filter_type.depth,
                            filter_type.num_filters,
                            [1, 1, 1, 1]))
            x = tf.concat(3, output_fmap)
            print(self.verbose)
            self.layer += 1

        with tf.variable_scope('sub_add'):
            in_filter = self.hps.layers[self.layer - 2].depth
            out_filter = self.hps.layers[self.layer - 1].tot_num_filters
            if (stride != [1]*4):
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            if (in_filter < out_filter):
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            elif (in_filter > out_filter) :
                
                orig_x = tf.nn.avg_pool(orig_x,
                        [1,1,1,3],
                        [1,1,1,3], 
                        'VALID')
                print(orig_x)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _plain(self, x, in_filter, out_filter, stride, filter_size,
               activate_before_residual=False, kernel_name='DW'):
        """Plain unit with 2 sub layers."""
        with tf.variable_scope('shared_activation'):
            x = self._batch_norm('init_bn', x)
            x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, kernel_name + str(self.layer), filter_size, in_filter, out_filter, stride)
            self.layer += 1

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = tf.nn.relu(x, name='relu')
            x = self._conv('conv2', x, kernel_name + str(self.layer), filter_size, out_filter, out_filter, [1, 1, 1, 1])
            self.layer += 1

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride, filter_size,
                             activate_before_residual=False):
        """Bottleneck resisual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = tf.nn.relu(x, name='relu')
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = tf.nn.relu(x, name='relu')
            x = self._conv('conv2', x, filter_size, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = tf.nn.relu(x, name='relu')
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _conv(self, name, x, kernel_name, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                kernel_name, [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            self.verbose += '|CONV[%d x %d x %d x %d]|'%(filter_size, filter_size, in_filters, out_filters)
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        print('\tFC[1 x %d]'%out_dim)
        return tf.nn.xw_plus_b(x, w, b)

    @staticmethod
    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _depth_avg_pool(x, s):
        x_t = tf.perm(x,[0,1,3,2])
        x_t_reduced = tf.nn.avg_pool(x,[1, 1, s, 1],[1, 1, s, 1],'VALID')
        return tf.perm(x,[0,1,3,2])

    @staticmethod
    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def __str__(self):
        return "Residual Networks Model"
