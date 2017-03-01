import imagenet_input
from sys import exit
import resnet_model
import tensorflow as tf
import argparse
from Imagenet import Imagenet
from Cifar import Cifar
from Mnist import Mnist
from resnet_model import ResNet
from lenet_model import Lenet
from regularizers import Regularizer,\
        WeightDecay, LayerWidth, NormalizedKernelSize, \
        KernelSize, Complexity, WeightedKernelSize, \
        KernelShape, Lasso, KernelSizeEnhanced,\
        KernelFeatureMap
from hyper_parameters import HParams
import utility as ut
parser = argparse.ArgumentParser(description='Load model for post-training computation')
parser.add_argument('--data_path', type=str, default='/data/ext/MNIST', help='Filename for training data.')
parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='Choose dataset',
                    choices=['cifar10', 'cifar100', 'imagenet', 'mnist'])
parser.add_argument('--res_mod', dest='res_mod', type=str, default='residual',
                    help='Type of network (Plain, residual or bottleneck')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.0,
                    help='Penalization for the sparsity regularization')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='Number of images per batch')
parser.add_argument('--log_root', dest='log_root', type=str, default='',
                    help='Directory to keep the checkpoints. Should be a parent directory of eval_dir')
parser.add_argument('--model_type', dest='model_type', type=str, default='resnet4-10',
                   help='Choose model depth and kernel size (model_depth-kernel_size)')
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--output_file", type=str)
parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--reg', dest='reg',action='append', help='Choose regularization')
parser.add_argument('--conf', type=str, default='')
args = parser.parse_args()

class ModelClass(Regularizer, ResNet): 
    def __init__(self, *args, **kwargs):
        Regularizer.__init__(self, *args, **kwargs)
        ResNet.__init__(self,*args, **kwargs)

#class ModelClass(Regularizer, Lenet): 
#    def __init__(self, *args, **kwargs):
#        Regularizer.__init__(self, *args, **kwargs)
#        Lenet.__init__(self,*args, **kwargs)

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
            add_reg = WeightDecay(alpha=0.001)
        elif reg == 'lasso':
            add_reg = Lasso(alpha=10**args.alpha)
        elif reg == 'fmap':
            add_reg = KernelFeatureMap(alpha=10**args.alpha)
        else :
            add_reg = WeightDecay(alpha=0.001)

        model.add_regularizer(add_reg)


def get_device():
    return args.device

def get_session():
    return sess

def get_alpha():
    return args.alpha

def get_model():
    return model

def get_model_type():
    return args.model_type

def get_output():
    return args.output_file

def get_hps():
    return hps

def get_dataset():
    return args.dataset
# Resnet hyper parameters
if (args.dataset == 'cifar10' or args.dataset == 'mnist'):
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100
elif args.dataset == 'imagenet':
    num_classes = 1001


hps = HParams(batch_size=args.batch_size,
              num_classes=num_classes,
              lrn_rate=0.1,
              res_mod=args.res_mod,
              optimizer=args.optimizer,
              dataset=args.dataset)

print("Loading Dataset ...")

# Load dataset
if args.dataset == 'mnist':
    dataset = Mnist(args.model_type, args.log_root)
if args.dataset.startswith('cifar'):
    dataset = Cifar(args.model_type, args.dataset, args.log_root)

# Build model given a specific dataset
if args.conf == '':
    print('Build model ' + args.model_type )
else:
    print('Build model from file ' + args.conf)
dataset.build_hps(args.data_path, hps, 'train', args.conf )
model = ModelClass(hps, dataset.images, dataset.labels, 'train',
        alpha=10**args.alpha)
add_regularizer(model)
print(model)

# Generate all needed partition for our model
ut.generate_all_partitions(hps)

# Build graph with all regularizers
model.build_graph()

# Session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.train.start_queue_runners(sess)

# Saver creation to load the checkpoint model
saver = tf.train.Saver()
try:
    ckpt_state = tf.train.get_checkpoint_state(args.log_root)
    print("Checkpoint found")
except tf.errors.OutOfRangeError as e:
    exit('Cannot restore checkpoint: %s'%  e)
if not (ckpt_state and ckpt_state.model_checkpoint_path):
    exit('No model to eval yet at %s' % args.log_root)
saver.restore(sess, ckpt_state.model_checkpoint_path)
print("Model Restored")
print("Loaded Model :\n"
      "\tType : " + args.res_mod + "\n"
      "\tPenalization alpha = " + str(args.alpha))
print('Variables :')
if args.verbose:
    for var in tf.trainable_variables():
        name = var.op.name
        sess.run(var.op.name)
        print("\t" + name)

