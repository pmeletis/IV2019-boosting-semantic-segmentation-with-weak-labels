"""
Input pipeline for Mapillary Vistas tfrecords created with KEYS2FEATURES_v5 specifications.

For training:
Includes preprocessing: change range to [-1, 1], resize to hf x wf, and augmentations
Vistas image dimensions are not the same across all images.
Vistas average size: (2481, 3419)
max training dims: (827, 1139) (= average/3.0), [TF r1.11, 4 GPUs, Nb=12 (3/GPU), psp, bootstrapping]
max training dims: (918, 1266) [TF r1.12, 4 GPUs, Nb=8 (2/GPU), psp, bootstrapping]
max training dims: (1240, 1710) (= average/2.0), [TF r1.12, 4 GPUs, Nb=4 (1/GPU), psp, cross-replica batch norm, no fov_* expansion and only bilinear upsampling]

For evaluation:
Fow now only basic resizing to network dimensions is implemented. In the future a better algorithm
using smart cropping and resizing (similar to training) will be implemented.
"""

import tensorflow as tf
import sys, glob
from os.path import join, split, realpath
import functools
sys.path.append(split(split(realpath(__file__))[0])[0])
from preprocessing import augmentation_library as augment
from PIL import Image
import numpy as np
from datetime import datetime
from input_pipelines.utils import from_0_1_to_m1_1, resize_images_and_labels, get_temp_Nb
from utils.utils import _replacevoids

# public functions:
#   train_input, evaluate_input, predict_input

SHUFFLE_BUFFER = 2000
NUM_PARALLEL_CALLS = 15

# Vistas KEYS2FEATURES_v5
KEYS2FEATURES = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'image/format':  tf.FixedLenFeature((), tf.string, default_value=b'jpeg'),
    'image/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'image/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'image/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'label/format':  tf.FixedLenFeature((), tf.string, default_value=b'png'),
    'label/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'label/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'label/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    }

def _parse_tfexample(example):
  """
  tfexample parsing using KEYS2FEATURES_v5.

  Arguments:
    example: tf.string, 0-D, a serialized tf.train.Example example

  Returns:
    image: tf.Tensor, tf.uint8, (?, ?, ?)
    label: tf.Tensor, tf.uint8, (?, ?)
    im_path: tf.Tensor, tf.string, ()
    la_path: tf.Tensor, tf.string, ()
  """

  ## parse
  features = tf.parse_single_example(example, KEYS2FEATURES)

  image = tf.image.decode_jpeg(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'])
  # label is decoded as a 3-D png image
  label = label[..., 0]
  im_path = features['image/path']
  la_path = features['label/path']

  return image, label, im_path, la_path

def _train_prebatch_processing(rim, rla, params):
  """
  TODO(panos): add more info...

  Arguments:
    rim: raw images, as extracted from tfrecords
    rla: raw labels, as extracted from tfrecords
    params: object with with the following attributes:
      height_feature_extractor: ...
      width_feature_extractor: ...
      training_lids2cids: ...
  """

  sfe = (params.height_feature_extractor, params.width_feature_extractor)

  ## prepare
  rim = tf.image.convert_image_dtype(rim, dtype=tf.float32)
  training_lids2cids = _replacevoids(params.training_problem_def['lids2cids'])
  rla = tf.gather(tf.cast(training_lids2cids, tf.int32), tf.to_int32(rla))

  ## preprocess
  rim.set_shape((None, None, None))
  rla.set_shape((None, None))
  proimages, prolabels = resize_images_and_labels(rim[tf.newaxis, ...],
                                                 rla[tf.newaxis, ...],
                                                 sfe,
                                                 preserve_aspect_ratio=params.preserve_aspect_ratio)
  proimage, prolabel = proimages[0], prolabels[0]

  # pre-batching augmentations
  pass

  return rim, rla, proimage, prolabel

def _train_parse_and_prebatch_processing(example, params):
  image, label, im_path, la_path = _parse_tfexample(example)
  # image = tf.Print(image, [tf.shape(image)], summarize=10)
  image, label, proimage, prolabel = _train_prebatch_processing(image, label, params)

  return image, label, proimage, prolabel, im_path, la_path

def _train_postbatching_processing(pims, plas, rips, rlps, config, params):

  # augmentation
  # random_X requires batch dimension (0) to be defined
  # pims.set_shape((get_temp_Nb(config, params.Nb), None, None, None))
  # plas.set_shape((get_temp_Nb(config, params.Nb), None, None))
  # pims = augment.random_color(pims)
  # pims = augment.random_blur(pims)
  # if labels are not horizontal flip equivariant then comment next line
  # pims, plas = augment.random_flipping(pims, plas)
  # training_lids2cids = _replacevoids(params.training_problem_def['lids2cids'])
  # pims, plas = augment.random_scaling(
  #     pims, plas, [1.0, 2.0], max(training_lids2cids))

  # center to [-1, 1)
  pims = from_0_1_to_m1_1(pims)

  return pims, plas, rips, rlps

def prebatch_dataset(config, params):
  del config

  dataset = tf.data.TFRecordDataset(params.tfrecords_path)
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(SHUFFLE_BUFFER))
  dataset = dataset.map(
      functools.partial(_train_parse_and_prebatch_processing, params=params),
      num_parallel_calls=NUM_PARALLEL_CALLS)

  return dataset

def postbatch_dataset(dataset, config, params):
  dataset = dataset.map(
      functools.partial(_train_postbatching_processing, config=config, params=params),
      num_parallel_calls=NUM_PARALLEL_CALLS)

  return dataset

def train_input(config, params):
  """
  Returns a tf.data.Dataset for training from Mapillary Vistas tfrecords.

  Arguments:
    config: unused for now
    params: object with with the following attributes:
      required by _train_parse_and_prebatch_processing:
        tfrecords_path: path of tfrecords using KEYS2FEATURES_v5
        training_lids2cids: ...
      required by this function
        Ntrain: number of tfexamples in tfrecords
        Nb: number of examples per batch

  Returns:
    A tf.data.Dataset dataset containing (features, labels) tuples where:
      features = {'proimages', 'rawimagespaths', 'rawlabelspaths'}
      labels = {'prolabels'}
  """

  def _grouping(pim, pla, imp, lap):
    # group dataset elements as required by estimator
    features = {
        # 'rawimages': tf.zeros_like(pim),
        'proimages': pim,
        'rawimagespaths': imp,
        'rawlabelspaths': lap,
        }
    labels = {
        # 'rawlabels': tf.zeros_like(pla),
        'prolabels': pla,
        }

    # next line for distributed debugging
    # tf.string tensors is not supported for DMA read/write to GPUs (TF bug)
    if params.distribute:
      del features['rawimagespaths']
      del features['rawlabelspaths']

    return (features, labels)

  with tf.variable_scope('input_pipeline'):
    dataset = prebatch_dataset(config, params)
    # Vistas raw images and labels don't have the same spatial size and cannot be batched
    dataset = dataset.map(lambda rim, rla, pim, pla, rip, rlp: (pim, pla, rip, rlp))
    dataset = dataset.batch(get_temp_Nb(config, params.Nb))
    dataset = postbatch_dataset(dataset, config, params)
    dataset = dataset.map(_grouping, num_parallel_calls=NUM_PARALLEL_CALLS)
    dataset = dataset.prefetch(None)

  return dataset

def _evaluate_preprocess(image, label, params):

  SIZE_FEATURE_EXTRACTOR = (params.height_feature_extractor, params.width_feature_extractor)

  ## prepare
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  evaluation_lids2cids = _replacevoids(params.evaluation_problem_def['lids2cids'])
  label = tf.gather(tf.cast(evaluation_lids2cids, tf.int32), tf.to_int32(label))

  ## preprocess
  proimage = tf.image.resize_images(image, SIZE_FEATURE_EXTRACTOR)
  prolabel = tf.image.resize_images(label[..., tf.newaxis],
                                    SIZE_FEATURE_EXTRACTOR,
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

  proimage = from_0_1_to_m1_1(proimage)

  print('debug: proimage, prolabel', proimage, prolabel)

  return image, label, proimage, prolabel

def _evaluate_parse_and_preprocess(example, params):
  image, label, im_path, la_path = _parse_tfexample(example)
  _, _, proimage, prolabel = _evaluate_preprocess(image, label, params)

  return proimage, prolabel, im_path, la_path

def evaluate_input(config, params):

  del config

  def _grouping(pim, pla, imp, lap):
    # group dataset elements as required by estimator
    features = {
        'proimages': pim,
        'rawimagespaths': imp,
        'rawlabelspaths': lap,
        }
    labels = {
        'prolabels': pla,
        }
    return (features, labels)

  dataset = tf.data.TFRecordDataset(params.tfrecords_path)
  dataset = dataset.map(
      functools.partial(_evaluate_parse_and_preprocess, params=params),
      num_parallel_calls=30)
  # IMPORTANT: if Nb > 1, then shape of dataset elements must be the same
  dataset = dataset.batch(params.Nb)
  dataset = dataset.map(_grouping, num_parallel_calls=30)
  dataset = dataset.prefetch(None)

  return dataset

def add_train_input_pipeline_arguments(argparser, ctx=None):
  """
  Add arguments required by the input pipeline.

  Arguments:
    argparser: an argparse.ArgumentParser object to add arguments
    ctx: a context object with a suffix_name attribute
  """

  context_name = ctx.suffix_name if ctx else ''
  argparser.add_argument('tfrecords_path' + ('_' + context_name if ctx else ''), type=str,
                         default='/media/panos/data/datasets/mapillary/mapillary-vistas-dataset_public_v1.0/tfrecords/train.tfrecord',
                         help='Training is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
  argparser.add_argument('--preserve_aspect_ratio' + ('_' + context_name if ctx else ''), action='store_true',
                         help='Resizes the input images respecting the aspect ratio using cropings.')

def add_evaluate_input_pipeline_arguments(argparser):
  """
  Add arguments required by the input pipeline.

  Arguments:
    argparser: an argparse.ArgumentParser object to add arguments
  """

  argparser.add_argument('tfrecords_path', type=str,
                         help='Evaluation is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
