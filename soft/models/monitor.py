import tensorflow as tf

TRESHES = [0.01, 0.001, 0.0001, 0.00001]

def count_kernel_tau(individual_l2, layer_nb):
   
    for thr in TRESHES:
        l2_norm_above = tf.cast(tf.greater_equal(individual_l2, thr), tf.int32)
        nb_non_empty_kernel = tf.reduce_sum(l2_norm_above)

        tf.scalar_summary('Layer ' +
                             str(layer_nb) +
                             '/#Kernel - tau ' +
                             str(thr),
                             nb_non_empty_kernel
                             )

def kernel_size_distribution(l2_norms_groups_layer, filter_size, layer_nb,
        nb_kernel):

    for thr in TRESHES:
        non_empty_groups = tf.cast(tf.greater_equal(l2_norms_groups_layer,thr),tf.float32)
        kernel_sizes = tf.div(tf.reduce_sum(tf.add_n(tf.split(0,filter_size,non_empty_groups))),float(nb_kernel))
        tf.histogram_summary('Layer %d/kernel sizes - tau %f' %(layer_nb, thr), kernel_sizes)
        tf.scalar_summary('Layer %d/AVG kernel sizes - tau %f' %(layer_nb, thr),
                tf.reduce_mean(kernel_sizes))
