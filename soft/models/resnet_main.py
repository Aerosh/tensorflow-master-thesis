"""ResNet Train/Eval module.
"""
import time
import cifar_input
import imagenet_input
import numpy as np
import resnet_model
import tensorflow as tf
import argparse
import sys
from Imagenet import Imagenet
from Cifar import Cifar
from Mnist import Mnist
from os import path
from utility import sanitize_lrn_rates, read_lrn_rates
from resnet_model import ResNet
from regularizers import Regularizer,\
        WeightDecay, LayerWidth, KernelSize,\
        WeightedKernelSize, NormalizedKernelSize, \
        Complexity, ComplexityBis,\
        KernelShape, Lasso, KernelSizeEnhanced, \
        KernelFeatureMap
from hyper_parameters import HParams
import utility as ut

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='Choose dataset',
                    choices=['cifar10', 'cifar100', 'imagenet', 'mnist'])
parser.add_argument('--mode', dest='mode', type=str, default='train', help='train or eval')
parser.add_argument('--data_path',  type=str, default='', help='Filename(s) for data.')
parser.add_argument('--image_size', dest='image_size', type=int, default=32, help='Image side length')
parser.add_argument('--train_dir', dest='train_dir', type=str, default='', help='Directory to keep training outputs')
parser.add_argument('--eval_batch_count', dest='eval_batch_count', type=int, default=500, help='Number of batch to eval')
parser.add_argument('--eval_once', dest='eval_once', action='store_true', help='Whether evaluate the model only once')
parser.add_argument('--log_root', dest='log_root', type=str, default='',
                    help='Directory to keep the checkpoints. Should be a parent directory of eval_dir')
parser.add_argument('--num_gpus', dest='num_gpus', type=int, default=1, help='Number of GPUs used for training ')
parser.add_argument('--alphas', dest='alpha', type=float, default=0.0,
                    help='Penalization for the sparsity regularization as power of 10')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='Number of images per batch')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0', help='GPU id for the training/eval')
parser.add_argument('--reg', dest='reg',action='append', help='Choose regularization')
parser.add_argument('--res_mod', dest='res_mod', type=str, default='residual', choices=['residual', 'plain'],
                    help='Type of network')
parser.add_argument('--zeroing', dest='zeroing', action='store', type=float, default=-1.0,
                    help='Zeroing empty kernel with L2-norm below a threshold')
parser.add_argument('--model_type', dest='model_type', type=str, default='',
                   help='Choose model depth and kernel size (model_depth-kernel_size)')
parser.add_argument('--lrn_rate_thrs',type=str, help="learning rate threshold. String format : 0\tthr1\thr3\tstop")
parser.add_argument('--lrn_rate_vals',type=str, help="learning rate values. String format : val1\tval2\tval3")
parser.add_argument('--eval_output_file', type=str, help='File output for evaluation final resut')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd','mom','adam'])
parser.add_argument('--conf', type=str, default='')
parser.add_argument('--zeroing_train', type=float, default=-1)

args = parser.parse_args()

ST_MSG = "\t [STATE MESSAGE] "

class ModelClass(Regularizer, ResNet): 
    def __init__(self, *args, **kwargs):
        Regularizer.__init__(self, *args, **kwargs)
        ResNet.__init__(self,*args, **kwargs)

def add_regularizer(model) :
    for reg in args.reg : 
        if reg == 'width':
            add_reg = LayerWidth(alpha=10**args.alpha)
        elif reg == 'size':
            add_reg = KernelSize(alpha=10**args.alpha)
        elif reg == 'size-en':
            add_reg = KernelSizeEnhanced(alpha=10**args.alpha)
        elif reg == 'n-size':
            add_reg = NormalizedKernelSize(alpha=10**args.alpha)
        elif reg == 'w-size':
            add_reg = WeightedKernelSize(alpha=10**args.alpha)
        elif reg == 'complexity':
            add_reg = Complexity(alpha=10**args.alpha)
        elif reg == 'complexitybis':
            add_reg = ComplexityBis(alpha=10**args.alpha)
        elif reg == 'shape' :
            add_reg = KernelShape(alpha=10**args.alpha)
        elif reg == 'wd' :
            add_reg = WeightDecay(alpha=10**args.alpha)
        elif reg == 'lasso':
            add_reg = Lasso(alpha=10**args.alpha)
        elif reg == 'fmap':
            add_reg = KernelFeatureMap(alpha=10**args.alpha)
        else :
            add_reg = WeightDecay(alpha=0.001)

        model.add_regularizer(add_reg)

def build_model(hps):
    """ Build model given the dataset""" 
    if args.mode == "train":
        lrn_rate_thrs, lrn_rate_vals, stop =  load_config_file()
    else:
        lrn_rate_thrs, lrn_rate_vals, stop = None,None,None
    
    # Build dataset Object
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        dataset = Cifar(args.model_type, args.dataset, args.log_root, lrn_rate_thrs,
            lrn_rate_vals, stop)
    elif args.dataset == 'imagenet':
        dataset = Imagenet(args.model_type, args.log_root, lrn_rate_thrs,
            lrn_rate_vals, stop)
    elif args.dataset == 'mnist':
        dataset = Mnist(args.model_type, args.log_root, lrn_rate_thrs,
            lrn_rate_vals, stop)

    # Build model given a specific dataset
    dataset.build_hps(args.data_path, hps, args.mode, args.conf)
    model = ModelClass(hps, dataset.images, dataset.labels, args.mode )
    add_regularizer(model)

    if args.zeroing_train >= 0.0:
        model.add_train_zeroing(KernelFeatureMap(alpha=10**args.alpha))
        model.add_train_zeroing(KernelSize(alpha=10**args.alpha))


    print(model)

    # Generate all needed partition for our model
    ut.generate_all_partitions(hps)

    # Build graph and regularization groups 
    model.build_graph()


    if args.mode.startswith("train"):
        train(model, dataset)
    elif args.mode.startswith("eval"):
        evaluate(model, dataset)


def load_config_file():
    if((args.lrn_rate_thrs is None) or (args.lrn_rate_vals is None)):
        print('Read')
        return read_lrn_rates(args.log_root + "/lrn_rate.txt", args.dataset)
    else :
        print('Sanitize')
        return sanitize_lrn_rates(args.lrn_rate_thrs, args.lrn_rate_vals)


def train(model, dataset):
    """Training loop."""
    check_op = tf.add_check_numerics_ops()

    if args.zeroing_train >= 0.0:
        zeroing_train = model.zeroing_train(args.zeroing_train)

    ## Root folders to store model/events
    summary_writer = tf.train.SummaryWriter(args.train_dir + '/_alpha_' + str(10**args.alpha))
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=args.log_root,
                             is_chief=True,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=300,
                             global_step=model.global_step)

    sess = sv.prepare_or_wait_for_session(config=config)

    # Setup of the learning rate and the starting step number
    step = sess.run(sv.global_step)


    lrn_rate = dataset.learning_rate(step)
    print("Start at step : " + str(step))
    print(ST_MSG + 'Start iterations')
    try:
        while not sv.should_stop():
            if(args.dataset != 'mnist'):
                (_, summaries, loss, predictions, truth, train_step) = sess.run(
                    [model.train_op, model.summaries, model.cost, model.predictions,
                     model.labels, model.global_step], 
                    feed_dict={model.lrn_rate: lrn_rate})

            else: 
                mnist_images, mnist_labels = dataset.next_batch()
                (_, summaries, loss, predictions, truth, train_step) = sess.run(
                    [model.train_op, model.summaries, model.cost, model.predictions,
                     model.labels, model.global_step], 
                    feed_dict={model.lrn_rate: lrn_rate, model._images : mnist_images,
                        model.labels : mnist_labels })


            if args.zeroing_train >= 0.0:
                sess.run(zeroing_train)
            lrn_rate = dataset.learning_rate(step)

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            precision = np.mean(truth == predictions)

            step += 1
            if step % 100 == 0:

                precision_summ = tf.Summary()
                precision_summ.value.add(
                    tag='Precision', simple_value=precision)
                summary_writer.add_summary(precision_summ, train_step)
                summary_writer.add_summary(summaries, train_step)
                tf.logging.info('loss: %.3f, precision: %.3f\n' % (loss, precision))
                summary_writer.flush()

            if dataset.stop(step):
                sv.stop()
                print(ST_MSG + "Training stopped")
                sv.saver.save(sess,sv.save_path + 'model.ckpt-' + str(step)) 
  
    except KeyboardInterrupt:
        sv.saver.save(sess,sv.save_path + 'model.ckpt-' + str(step)) 
        print(ST_MSG + "Interrupted at iteration " + str(step))

def evaluate(model, dataset):
    """Eval loop."""

    print(ST_MSG + " Start Evaluation")

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.train.start_queue_runners(sess)

    if args.zeroing >= 0.0:
        zeroing = model.zeroing(args.zeroing)
        sess.run(tf.initialize_all_variables())

    top1_total = 0.0
    top5_total = 0.0

    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(args.log_root)
        except tf.errors.OutOfRangeError as e:
            exit('Cannot restore checkpoint: %s'% e)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            exit('No model to eval yet at %s'% args.log_root)
        print('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        print(ST_MSG + "Model Restored")

        if args.zeroing >= 0.0:
            sess.run(zeroing)

        step = 1
        start_time = time.time()
        for it in range(args.eval_batch_count):
            if args.dataset != "mnist":
                (summaries, loss, predictions, truth, train_step, top1, top5, images) = sess.run(
                    [model.summaries, model.cost, model.predictions,
                     model.labels, model.global_step, model.top1, model.top5, model._images])
            else:
                mnist_images, mnist_labels = dataset.next_batch()
                (summaries, loss, predictions, truth, train_step, top1, top5, images) = sess.run(
                    [model.summaries, model.cost, model.predictions,
                     model.labels, model.global_step, model.top1, model.top5,
                     model._images],
                    feed_dict={model._images : mnist_images, model.labels :
                        mnist_labels})

            top1_total += np.sum(top1)
            top5_total += np.sum(top5)

            if it % 10 == 0:
                print("Step " + str(step - 1) + ": Top-1 : " + str(top1_total/float(step*args.batch_size)) +
                      " - Top-5 : " + str(top5_total/float(step*args.batch_size)))

            step += 1

        top1_acc = top1_total/float(args.eval_batch_count*args.batch_size)
        top5_acc = top5_total/float(args.eval_batch_count*args.batch_size)

        elapsed_time = time.time() - start_time

        print("Top-1 : " + str(top1_acc))
        print("Top-5 : " + str(top5_acc))
        print("Evaluation time : " + str(elapsed_time))
        
        with open(args.eval_output_file, 'a') as file:
            file.write("%1.5g\t%1.5g\t%1.5g\t%1.5g\n" %
                    (args.alpha, args.zeroing,  top1_acc, top5_acc))
            file.flush()
            file.close()
        if args.eval_once:
            break


def main(_):
    if args.num_gpus == 0 or args.num_gpus > 1:
        dev = '/cpu:0'
    elif args.num_gpus == 1:
        dev = '/gpu:' + args.gpu_id
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if args.train_dir == '':
        args.train_dir = args.log_root

    if args.dataset == 'cifar10' or args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1001

    hps = HParams(batch_size=args.batch_size,
                  num_classes=num_classes,
                  lrn_rate=0.1,
                  res_mod=args.res_mod,
                  optimizer='adam',
                  dataset=args.dataset)

    with tf.device(dev):
        build_model(hps)
if __name__ == '__main__':
    tf.app.run()
