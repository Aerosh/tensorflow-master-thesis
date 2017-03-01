import tensorflow as tf
from os import path
import numpy as np

ring_partitions = []
fmap_partitions = []
shape_partitions = []

class LayerPartitions(object):

    def __init__(self, partitions):
        self.partitions = partitions

    def expand(self, dim):
        """
        Expand dimension on all partitions matrix for all available kernsel
        sizes
        """
        partitions_expanded = []
        for partition in self.partitions:
            partitions_expanded.append(partition.expand(dim))

        return LayerPartitions(partitions_expanded)

    def get_partition_matrix(self, filter_size):
        return  next((x.matrix for x in self.partitions if x.filter_size == filter_size))

class Partition(object):

    def __init__(self, matrix, layer_number, filter_size):
        self.matrix = matrix
        self.layer_number = layer_number
        self.filter_size = filter_size
   
    def expand(self, dim):
        """
        Expand dimension of partition matrix
        """
        matrix = map(lambda x : tf.expand_dims(x, dim), self.matrix)
        return Partition(matrix, self.layer_number, self.filter_size)

    def __repr__(self):
        return 'matrix : some matrix - layer #' + str(self.layer_number) + ' -  filter size : ' + str(self.filter_size)

def generate_all_partitions(hps):
    """ Produce partition matrices for all the available regularization schemes"""
    global ring_partitions
    global shape_partitions
    r_pad = [
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]]
            ]

    s_pad = [
            [[0, 1], [0, 0]],
            [[1, 0], [0, 0]]
            ]

    for i, layer in enumerate(hps.layers):
        ring_part  = []
        shape_part = []
        for filter_type in layer.filters :
            filter_size = filter_type.filter_size
            nb_kernel = filter_type.num_filters
            ring_part.append(Partition(
                map(lambda x : tf.expand_dims(x,2),
                    generate_partitions(filter_size, nb_kernel,r_pad)),
                i, filter_size))
            shape_part.append(Partition(generate_partitions(filter_size,
                nb_kernel,s_pad), i, filter_size)) 
        ring_partitions.append(LayerPartitions(ring_part))
        shape_partitions.append(LayerPartitions(shape_part))
    
def generate_partitions(size, nb_kernel, paddings):
    """ Generate a partition matrix in accord to nested rings, shape or fmap in kernels

    e.g : Partition matrix for a 3x3 kernel for rings :
    [[0, 0, 0],
     [0, 2, 1],
     [0, 1, 1]]

    Shape partition for 6x6 kernel 
    [5, 3, 1, 0, 2, 4] for row and column
    """
    start = np.zeros((1,1))
    parts = tf.convert_to_tensor(start, dtype=tf.int32)
    for i in range(size - 1):
        parts += 1
        if i % 2 == 0:
            parts = tf.pad(parts, paddings[0])
        else:
            parts = tf.pad(parts, paddings[1])

    partitions = [ tf.cast(
                            tf.logical_and(tf.greater_equal(parts,0),tf.less_equal(parts,i)),
                        tf.float32)
                     for i in range(size)
                 ]
    return partitions

def generate_zeroing_matrix(zeroing_matrix, height, width,depth):

    for s in range(1, max(width, height, depth)):
        if s < width:
            zeroing_matrix = tf.pad(zeroing_matrix, [[0, 0], [0, 1], [0, 0], [0, 0]], "SYMMETRIC")
        if s < height:
            zeroing_matrix = tf.pad(zeroing_matrix, [[0, 1], [0, 0], [0, 0], [0, 0]], "SYMMETRIC")
        if s < depth:
            zeroing_matrix = tf.pad(zeroing_matrix, [[0, 0], [0, 0], [0, 1], [0, 0]], "SYMMETRIC")

    return zeroing_matrix

def n_dim_zeroing(n,indicator, height, width, depth):
    """ Expand dimensions pf indicator n times before generating some zeroing
    matrix """
    for _ in range(n):
        indicator = tf.expand_dims(indicator, 0)
    return generate_zeroing_matrix(indicator, height, width, depth)
    

def generate_groups_zeroing_matrix(indicators,size, depth, nb_kernel, layer_nb,
        group_partitions):
    """ Generate tensors to zero out nested ring groups below a given
    threhsold"""
    zeroing_matrix = tf.convert_to_tensor(np.zeros((size,size, depth,
        nb_kernel)), tf.float32)
    for i,ind in enumerate(indicators):
        if i == 0:
            continue
        if i == size - 1:
            continue
        indicator_tensor = n_dim_zeroing(3, ind, size, size, depth)
        group_zeroing = tf.expand_dims(tf.cast(
                    tf.equal(group_partitions[layer_nb].get_partition_matrix(size)[size - i -1],0),
                    tf.float32),
                    3)
        zeroing_matrix += tf.mul(group_zeroing,indicator_tensor)
    zeroing_matrix += n_dim_zeroing(3,ind, size, size, depth)
    return zeroing_matrix


def kernel_l2_loss(layer):
    return tf.sqrt(tf.reduce_sum(tf.square(layer), [0,1,2]))


def extract_filter_prop(var_name):
    """ Extract filter size from variable name
    """
    layer_name = var_name.split('DW')[-1]
    return map(lambda x: int(x), layer_name.split('-'))


def sanitize_lrn_rates(thrs_str, vals_str):
    lrn_thrs = [int(thr) for thr in thrs_str.split('\t')]
    lrn_vals = [float(val) for val in vals_str.split('\t')]

    return lrn_thrs[0:-1], lrn_vals, lrn_thrs[-1]


def read_lrn_rates(fpath, dataset):
    if path.isfile(fpath):
        f = open(fpath, 'r')
    else :
        print('\t[STATE MSG] Learning rate not specified. Take default values')
        f = open('default_values/' + dataset + '_lrn_rates.txt')

    thrs_thrsh = f.readline()[0:-1]
    thrs_vals = f.readline()

    return sanitize_lrn_rates(thrs_thrsh, thrs_vals)
