import tensorflow as tf
#import matplotlib.pyplot as plt
import load_model
import numpy as np
import complexity
import utility as ut
from os import makedirs, path
class LayerGroupsL2Norms(object):

    def __init__(self, groups_l2_norms, filter_size, layer_nb, *args, **kwargs):
        super(LayerGroupsL2Norms, self).__init__(*args, **kwargs)
        self.groups_l2_norms = groups_l2_norms
        self.layer_nb = layer_nb
        self.filter_size = filter_size


if load_model.get_model_type().startswith('resnet8'):
    in_map_szs = [112] + [56]*2 + [28]*2 + [14]*2 + [1]
elif load_model.get_model_type().startswith("resnet20"):
    in_map_szs = [32]*7 + [16]*6 + [8]*6 + [1]
elif load_model.get_model_type().startswith("resnet18"):
    in_map_szs = [112] + [56]*4 + [28]*4 + [14]*4 + [7]*4 + [1]
elif  load_model.get_model_type().startswith("resnet4"):
    in_map_szs = [28]*3
elif  load_model.get_model_type().startswith("resnet10"):
    in_map_szs = [28]*5 + [14]*4 + [1]
elif  load_model.get_model_type().startswith("lenet"):
    in_map_szs = [32, 28, 14, 10, 5, 1, 1, 1]
taus = [0] + [10**(-(5 - i)) for i in range(0, 5)]
print "Start retrieve infos"

if load_model.args.dataset == 'mnist':
    n_channel = 1
else:
    n_channel = 3

def to_tuple(sizes):
    output = []
    for row,col in zip(sizes[0],sizes[1]):
        stack = []
        for i in range(len(row)):
            stack.append((row[i],col[i]))
        output.append(stack)
    return output

def write_dist(sizes_dist, file_pointer, alpha):
    global taus
    size_dist_t = np.transpose(sizes_dist,(1,2,0))
    # Kernel size distribution for a given tau
    string_format = "%d\t%d\t%1.2f" + "\t%d"*len(taus) + "\n"
    for i,layer in enumerate(size_dist_t):
        for j,size in enumerate(layer):
            pattern = (i,j,alpha)
            for tau in size:
                pattern += (tau,)
            file_pointer.write(string_format%pattern)
            file_pointer.flush()
    file_pointer.close()

def write_confusion_matrix(sizes_tuple, root, alpha, tau):
    sij_hat = 5
    sij_hat += 1
    conf_mat_tot = np.zeros((sij_hat, sij_hat))
    conf_mat_temp = np.zeros((sij_hat, sij_hat)) 
    for l in sizes_tuple:
        for k in l:
            conf_mat_temp[k[0],k[1]] += 1
            
        conf_mat_tot = conf_mat_tot + conf_mat_temp
        conf_mat_temp = np.zeros((sij_hat, sij_hat)) 

    np.save(root + 'alpha-' + str(alpha) + '-tau-' + str(tau),
            conf_mat_tot) 


def kernel_size_single_layer(l2norms, tau):
    return tf.add_n(tf.split(0, l2norms.filter_size,
        tf.cast(tf.greater_equal(l2norms.groups_l2_norms,tau),tf.int32)))

def filter_empty_depth_kernels(sizes, depths):
    sizes_corr = []
    depths = [depths[0][0]] + [np.sum(l,0) for l in depths[1:]]
    layer_nb = 0
    for layer_s,layer_d in zip(sizes, depths):
        layer_sizes = []
        for k_d,k_s in zip(layer_d, layer_s):
            if k_d > 0:
                layer_sizes.append(k_s)
            else:
                layer_sizes.append(0)
        layer_nb += 1
        sizes_corr.append(np.array(layer_sizes))

    return  np.array(sizes_corr)
                   
def computer_kernel_per_layer(layers_l2norm, tau):

    sizes_tensors = []
    
    for layer_l2norm in layers_l2norm :
        sizes_tensors.append(
                tf.cast(
                    tf.greater_equal(layer_l2norm.groups_l2_norms, tau)
                    , tf.int32)*layer_l2norm.filter_size
                )

    sizes = sess.run(sizes_tensors)
    return sizes,sizes

def compute_kernel_sizes_rings(layers_groups_l2norms, tau, depths):
    sizes_tensors = []
    layer_nb = 0
    same_layer = []
    for group_l2norms in layers_groups_l2norms:
        if group_l2norms.layer_nb != layer_nb:
            sizes_tensors.append(tf.concat(0,same_layer))
            same_layer = []
            layer_nb = group_l2norms.layer_nb

        same_layer.append(kernel_size_single_layer(group_l2norms, tau))
    sizes_tensors.append(tf.concat(0,same_layer)) 
    sizes = sess.run(sizes_tensors)
    sizes = filter_empty_depth_kernels(sizes,depths)
    return sizes, sizes

    
def compute_kernel_sizes_cr(layers_groups_l2norms, tau):
    sizes_cols = []
    sizes_rows = []
    layers_groups_l2norms = np.transpose(layers_groups_l2norms).tolist()
    for (col_l2norm, row_l2norm) in zip(layers_groups_l2norms[0],layers_groups_l2norms[1]):
        sizes_cols.append(kernel_size_single_layer(col_l2norm, tau))
        sizes_rows.append(kernel_size_single_layer(row_l2norm, tau))

    sizes_col = sess.run(sizes_cols)
    sizes_row = sess.run(sizes_rows)

    return sizes_row, sizes_col

def valid_depth(kernels_depth, tau):
    valid_depths = []
    for d in kernels_depth:
        depth = tf.cast(tf.greater_equal(d, tau), tf.int32)
        valid_depths.append(sess.run(depth))

    return valid_depths

with tf.device('/' + load_model.get_device() +':0'):

    # Load useful component from the interface
    model = load_model.model
    sess = load_model.sess
    hps = load_model.hps
    alpha = load_model.args.alpha
    # Definition of gathering arrays
    layers_groups_l2norms = []
    layers_depths_l2norms = []
    sizes_dist = []

    # Written file 
    if not any('shape' in reg for reg in load_model.args.reg):
        file_kernel_size = open(load_model.args.output_file + "-kernel_szs.dat",'a')
    else:
        if not path.exists('results/conf-mat/' + load_model.args.output_file):
            makedirs('results/conf-mat/' + load_model.args.output_file)
        root = 'results/conf-mat/' + load_model.args.output_file + '/'
    file_complexity = open(load_model.args.output_file + "-complexity.txt",'a')

    for var in tf.trainable_variables():
        if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
            layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
            
            # Compute l2 norms
            if any('size' in reg or 'complexity' in reg for reg in load_model.args.reg):
                layers_groups_l2norms.append(
                        LayerGroupsL2Norms(model._groups_l2_norm(var,
                    layer_nb, filter_size), filter_size, layer_nb))

            elif any('width' in reg for reg in load_model.args.reg):
                layers_groups_l2norms.append(                     
                        LayerGroupsL2Norms(model._groups_l2_norm(var),
                            filter_size, layer_nb))
            elif any('shape' in reg for reg in load_model.args.reg):
                layers_groups_l2norms.append(
                    [LayerGroupsL2Norms(model._groups_l2_norm(var, 'col'),
                        filter_size,layer_nb),
                        LayerGroupsL2Norms(model._groups_l2_norm(var,'row'),
                            filter_size, layer_nb)])
                
            layers_depths_l2norms.append(model.depth_l2norms(var))
                

    for tau in taus:

        # Computer number of non_zero feature map for each kernels
        depths = valid_depth(layers_depths_l2norms, tau)

        # Retrieve size by taking into account groups construction
        if any('size' in reg or 'complexity' in reg for reg in load_model.args.reg):
            sizes = compute_kernel_sizes_rings(layers_groups_l2norms, tau, depths)
        elif any('width' in reg for reg in load_model.args.reg):
            sizes = computer_kernel_per_layer(layers_groups_l2norms, tau)
        elif any('shape' in reg for reg in load_model.args.reg):
            sizes = compute_kernel_sizes_cr(layers_groups_l2norms, tau)
        
        sizes_dist.append([
            [np.sum(size == i) for i in range(hps.layers[j].get_max_size() + 1)]
            for j,size in enumerate(sizes[0])])
        
        sizes_tuple = to_tuple(sizes)
        file_complexity.write('%1.5g\t%1.5g\t%1.5g\t\n'%
                (load_model.get_alpha(), tau,
                    complexity.complexity(in_map_szs, sizes_tuple, n_channel, depths)))

        if any('shape' in reg for reg in load_model.args.reg):
            write_confusion_matrix(sizes_tuple, root, alpha, tau)

    if any('size' in  reg or 
            'complexity' in reg or 
            'width' in reg
            for reg in load_model.args.reg):
        write_dist(sizes_dist, file_kernel_size,alpha)

