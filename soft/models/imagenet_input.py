""" Handle the inputs reading, preprocessing and split into batches"""
import os
from random import randint

import tensorflow as tf

PATCH_SIZE = 256
RANDOM_PATCH_SIZE = 224
flags = tf.app.flags
FLAGS = flags.FLAGS
SINGLE_CROP = True

flags.DEFINE_integer('num_classes', 1001, 'Number of categories in Imagenet')
flags.DEFINE_integer('num_thread', 8, 'Number of threads used')
flags.DEFINE_integer('num_reader', 8, 'Number of readers used')

# Images are processed asynchronously with multiple reader.
# Ensuring some good performance in reading but also mixing accross the examples
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16, 'Size of the queue of preprocessed images.')


def read_and_decode(serialized_example):
    """ Read the TFRecords and decode each elements"""
    # Read the image and take the wanted parameters
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=1),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=1)
        }
    )

    # Decode the jpeg image and apply some cropping
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)

    image = tf.reshape(image, tf.pack([height, width, 3]))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image, label, features['image/class/text'], height, width


def _compute_longer_edge(height, width, new_shorter_edge):
    return tf.cast(width * new_shorter_edge / height, tf.int32)


def ten_crops_patches(image):
    boxes = [
        [0, 0, 0.875, 0.875],
        [0, 0.125, 0.875, 1],
        [0.125, 0, 1, 0.875],
        [0.125, 0.125, 1, 1],
        [0.0625, 0.0675, 0.9375, 0.9375],
        [0, 0.875, 0.875, 0],
        [0, 1, 0.875, 0.125],
        [0.125, 0.875, 1, 0],
        [0.125, 1, 1, 0.125],
        [0.0675, 0.9375, 0.9375, 0.0675]
    ]
    boxes_tensor = tf.convert_to_tensor(boxes, tf.float32)
    crop_size = tf.convert_to_tensor([RANDOM_PATCH_SIZE, RANDOM_PATCH_SIZE], tf.int32)
    boxes_ind = tf.convert_to_tensor([0]*10, tf.int32)

    return tf.image.crop_and_resize(tf.expand_dims(image, 0), boxes_tensor, boxes_ind, crop_size)


def distort_image(image, height, width, train):
    """Distort the input image for the data augmentation"""

    # Resize the image according to the shorter edge
    if train:
        new_shorter_edge = tf.constant(randint(256, 480), dtype=tf.int32)
    else:
        new_shorter_edge = tf.constant(256, dtype=tf.int32)

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = tf.cond(height_smaller_than_width,
                                   lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
                                   lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
                                   )

    if tf.__version__ == "0.11.0rc0":
        image = tf.image.resize_images(image, new_height_and_width)
    else:
        image = tf.image.resize_images(image, new_height_and_width[0], new_height_and_width[1])

    if train:
        # Randomly choose between cropping the image or its horizontal flip
        image = tf.random_crop(tf.image.random_flip_left_right(image),
                               [RANDOM_PATCH_SIZE, RANDOM_PATCH_SIZE, 3])

    else:
        # Crop the image to 256*256
        image = tf.image.crop_to_bounding_box(image, 0, 0, PATCH_SIZE, PATCH_SIZE)
        if SINGLE_CROP:
            image = tf.image.crop_to_bounding_box(image, (PATCH_SIZE - RANDOM_PATCH_SIZE)/2,
                                                  (PATCH_SIZE - RANDOM_PATCH_SIZE)/2,
                                                  RANDOM_PATCH_SIZE, RANDOM_PATCH_SIZE)

    image = tf.image.per_image_whitening(image)

    return image


def batch_inputs(train, batch_size, pattern, data_path):
    """ Generate batch for CNN network and preprocess in the case of training batches"""

    filenames = tf.gfile.Glob(os.path.join(data_path, '%s-*' % pattern))

    # Create filename_queue
    if train:
        filename_queue = tf.train.string_input_producer(
            filenames,
            shuffle=True,
            capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(
            filenames,
            shuffle=False,
            capacity=1)

    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    # Create examples queue
    if train:
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])

    reader = tf.TFRecordReader()
    if FLAGS.num_reader > 1:
        enqueue_ops = []
        for _ in range(FLAGS.num_reader):
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        serialized_example = examples_queue.dequeue()

    else:
        _, serialized_example = reader.read(filename_queue)

    images_and_labels = []

    for thread in range(FLAGS.num_thread):
        image, label, human_label, height, width = read_and_decode(serialized_example)
        image = distort_image(image, height, width, train)
        if train or (not train and SINGLE_CROP):
            images_and_labels.append([image, label])
        else:
            images_and_labels.append([ten_crops_patches(image), label])

    if not train and not SINGLE_CROP:
         batch_size /= 10

    images, labels = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=2 * FLAGS.num_thread * batch_size)

    if train or (not train and SINGLE_CROP):
        labels = tf.reshape(labels, [batch_size, 1])
    else:
        batch_size *= 10
        labels = tf.reshape(tf.tile(labels, tf.convert_to_tensor([10], tf.int32)), [batch_size, 1])

    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(1, [indices, labels]),
        [batch_size, FLAGS.num_classes], 1.0, 0.0)

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, RANDOM_PATCH_SIZE, RANDOM_PATCH_SIZE, 3])

    tf.image_summary('images', images)

    return images, labels


def inputs(batch_size, data_path):
    """ Generate batches of regular inputs for the validation"""
    return batch_inputs(False, batch_size, 'validation', data_path)


def distorded_input(batch_size, data_path):
    """ Generate batches of distorded inputs for the training"""
    return batch_inputs(True, batch_size, 'train', data_path)
