import tensorflow as tf
import load_model
import numpy as np
import utility as ut

def ind_l2_loss(layer):

    layer_square = tf.square(layer)
    layer_sum_square = tf.reduce_sum(layer_square, [0, 1, 2])
    return tf.sqrt(layer_sum_square)


def plot_hist(data, title, fname):
    plt.hist(l2_losses, bins=np.arange(min(data), max(data) + max(data) / 50.0, max(data) / 50.0))
    plt.title(title)
    plt.savefig(fname)
    plt.gcf().clear()


def count_kernel_tau(l2_losses, stack, tau):
    stack.append(np.sum(np.array(l2_losses) > tau))
    return stack

sess = load_model.sess
model = load_model.model
layers_groups_l2norms = []

for var in tf.trainable_variables():
    if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
        layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
        
        # Compute l2 norms
        if any('size' in reg for reg in load_model.args.reg ):
            layers_groups = model._groups_l2_norm(var, layer_nb, filter_size)

        elif any('lasso' in reg for reg in load_model.args.reg):
            layers_groups_l2norms.append(tf.reshape(var,[-1]))
       
        layers_l2 = sess.run(layers_groups)
        np.save("results/groups-l2-losses/l-0.5/resnet10-size-en-adam_e-2.0/" + load_model.args.output_file +
                "_l_ " + str(layer_nb) + "_e" + str( load_model.args.alpha),
                layers_l2)



#l2_losses = sess.run(tf.concat(0,layers_groups_l2norms))

#title = load_model.args.model_type + "-" + load_model.args.dataset + "-" + load_model.args.reg + "-" + load_model.args.optimizer + "-" + str(load_model.args.alpha)
#
#plot_hist(l2_losses,
#    title,
#    'results/plots/histogramms/' + title + ".png")
#
#np.save("results/groups-l2-losses/l21/" + load_model.args.output_file + "_e" +
#        str( load_model.args.alpha), l2_losses)

#count_kernel_tau(l2_losses, nb_kernel, 0.001)

