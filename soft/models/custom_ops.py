import tensorflow as tf
from tensorflow.python.framework import function

@function.Defun(tf.float32,tf.float32)
def l21_norm_grad(group, dloss):
    #thresh = tf.constant(1e-6, tf.float32)
    #condition = tf.greater_equal(group, thresh)
    #sub_gradient = tf.cond(condition,
    #        lambda: tf.constant(1.0), lambda: tf.constant(2.0))
    
    #return sub_gradient
    return tf.sigmoid(group)

@function.Defun(tf.float32, grad_func=l21_norm_grad)
def l21_norm_per_kernel(layer):
    """ Compute the l21 norm for each kernel in a layer
    The layer should already be squared and summed over the third dimension """
    l21_loss_per_layer = tf.sqrt(tf.reduce_sum(tf.reduce_sum(layer,0),0))
    print(l21_loss_per_layer)
    return l21_loss_per_layer
