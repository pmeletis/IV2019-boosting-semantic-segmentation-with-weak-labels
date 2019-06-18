"""
Input pipeline for Open Images v4 bounding boxes subset. For training and evaluation
a tf.data.Dataset is generated from the {train, val}-imageid2bboxes.p, using a tf.py_func for generating ground truth.
In this case we let TF parallelize GT generation instead of a creation with a serial generator. Time is reduced by 2.
The Python generator emmits imageid s and a list of (class, bbox_coords).

Open Images v4 bounding boxes subset average image size: (?, ?).
"""

import json
import os.path as op
import operator
import pickle
import pprint

import tensorflow as tf
import sys, glob
from os.path import join, split, realpath
import functools
sys.path.append(split(split(realpath(__file__))[0])[0])
# from preprocessing import augmentation_library as augment
from PIL import Image
import numpy as np
from datetime import datetime
from input_pipelines.utils import from_0_1_to_m1_1, resize_images_and_labels, get_temp_Nb
from utils.utils import _replacevoids

# public functions:
#   train_input, evaluate_input, predict_input

SHUFFLE_BUFFER = 2000
NUM_PARALLEL_CALLS = 15
MAX_N_BBOXES = 516

PATH_train = '/media/panos/data/datasets/open_images_v4/subset_bboxes/train'
FILEPATH_train_imageid2bboxes = '/media/panos/data/datasets/open_images_v4/subset_street_scenes_v1/train-imageid2bboxes.p'

mid2cid = {'/m/0199g':   0,  # bicycle
           '/m/01bjv':   1,  # bus
           '/m/0k4j':    2,  # car
           '/m/04_sv':   3,  # motorcycle
           '/m/07jdr':   4,  # train
           '/m/07r04':   5,  # truck
           '/m/01g317':  6,  # human (person originally but may include also rider)
           '/m/04yx4':   7,  # man
           '/m/03bt1vf': 8,  # woman
           '/m/01bl7v':  9,  # boy
           '/m/05r655': 10,  # girl
           '/m/015qff': 11,  # traffic light
           '/m/01mqdt': 12,  # traffic sign
           '/m/02pv19': 13,  # stop sign
           'void':      14,
          }

def _imageid_and_bboxes_generator(filepath):
  """outputs an image (np.float32, 3D) and the
     generated weak labels from bounding boxes (np.float32, 3D)
  """
  with open(filepath, 'rb') as fp:
    imageid2bboxes = pickle.load(fp)
  
  for imageid, bboxes in imageid2bboxes.items():
    Np = MAX_N_BBOXES - len(bboxes)
    if Np < 0:
      tf.logging.warn(f'Np = {Np}.')
    # get mid s and pad to MAX_N_BBOXES
    mids = [bbox[0].encode('utf-8') for bbox in bboxes]
    mids.extend([''.encode('utf-8')]*Np)
    # get coord s and pad to MAX_N_BBOXES
    coords = [bbox[1] for bbox in bboxes]
    coords.extend([(0., 1., 0., 1.)]*Np)
    yield imageid.encode('utf-8'), mids, coords, Np

def _generate_rla(imageid, mids, coords_normalized, rim_size):
  # mids: binary np.string, (Nbboxes)
  # coords_normalized: np.float32, (Nbboxes, 4)
  # rim_size: np.int32, (2,)
  rla = np.zeros((*rim_size, len(mid2cid)), dtype=np.float32)
  h, w = rim_size
  for mid, coords_normalized in zip(mids, coords_normalized):
    mid = mid.decode('utf-8')
    if mid in mid2cid.keys():
      xmin, xmax, ymin, ymax = map(int, map(operator.mul, coords_normalized, [w, w, h, h]))
      rla[ymin:ymax+1, xmin:xmax+1, mid2cid[mid]] += 1
  # per-pixel normalize rla to a dense multinomial distribution
  reduced_sum = np.sum(rla, axis=2, keepdims=True)
  # the following formula addresses the cases:
  #   [car, bus, person] per pixel
  #   [1, 0, 0]  -> [1, 0, 0]
  #   [2, 0, 0]  -> [1, 0, 0]
  #   [1, 1, 0]  -> [1/2, 1/2, 0]
  #   [2, 1, 0]  -> [2/3, 1/3, 0]
  rla = np.where(reduced_sum > 0.5,
                 rla / reduced_sum,
                 np.concatenate([np.zeros((len(mid2cid)-1), dtype=np.float32), np.ones((1,), dtype=np.float32)]))
  # assert np.all(np.abs(np.sum(rla, axis=2) - np.ones(rla.shape[:2], dtype=np.float32)) < 0.001), (
  #     f'some pixels in rla for imageid {imageid} doesn\'t represent a multinomial distribution.')
  return rla

def _train_prebatch_processing(imageid, mids, coords, Np, params):
  path = tf.strings.join([PATH_train, tf.strings.join([imageid, '.jpg'])], separator='/')
  rim = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
  rim = tf.image.convert_image_dtype(rim, tf.float32)

  mids = mids[:-Np]
  coords = coords[:-Np]

  rla = tf.py_func(_generate_rla, [imageid, mids, coords, tf.shape(rim)[:2]], tf.float32, stateful=False)
  rla.set_shape((None, None, None))

  sfe = (params.height_feature_extractor, params.width_feature_extractor)

  ## prepare
  pass

  ## preprocess
  proimage, prolabel = resize_images_and_labels(rim[tf.newaxis, ...],
                                                rla[tf.newaxis, ...],
                                                sfe,
                                                preserve_aspect_ratio=params.preserve_aspect_ratio)
  proimage, prolabel = proimage[0], prolabel[0]

  # pre-batching augmentations
  pass

  return proimage, prolabel, imageid

def _train_postbatching_processing(pims, plas, imageids, config, params):

  # augmentation
  # random_X requires batch dimension (0) to be defined
  # pims.set_shape((get_temp_Nb(config, params.Nb), None, None, None))
  # plas.set_shape((get_temp_Nb(config, params.Nb), None, None))
  #pims = augment.random_color(pims)
  #pims = augment.random_blur(pims)
  # pims, plas = augment.random_flipping(pims, plas)
  # training_lids2cids = _replacevoids(params.training_problem_def['lids2cids'])
  # pims, plas = augment.random_scaling(
  #     pims, plas, [1.0, 2.0], max(training_lids2cids))

  # center to [-1, 1)
  pims = from_0_1_to_m1_1(pims)

  return pims, plas, imageids

def prebatch_dataset(config, params):
  dataset = tf.data.Dataset.from_generator(
      functools.partial(_imageid_and_bboxes_generator, FILEPATH_train_imageid2bboxes),
      (tf.string, tf.string, tf.float32, tf.int32),
      output_shapes=((), (None,), (None, None), ()))
  dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(SHUFFLE_BUFFER))
  dataset = dataset.map(
      functools.partial(_train_prebatch_processing, params=params),
      num_parallel_calls=NUM_PARALLEL_CALLS)

  return dataset

def postbatch_dataset(dataset, config, params):
  dataset = dataset.map(
      functools.partial(_train_postbatching_processing, config=config, params=params),
      num_parallel_calls=NUM_PARALLEL_CALLS)

  return dataset

def train_input(config, params):
  """
  Returns a tf.data.Dataset for training from Open Images data
  """

  def _grouping(pim, pla, iid):
    # group dataset elements as required by estimator
    features = {
        # 'rawimages': rim,
        'proimages': pim,
        'imageids': iid,
        # 'rawimagespaths': imp,
        # 'rawlabelspaths': lap,
        }
    labels = {
        # 'rawlabels': rla,
        'prolabels': pla,
        }

    # next line for distributed debugging
    # tf.string tensors is not supported for DMA read/write to GPUs (TF bug)
    if params.distribute:
      # del features['rawimagespaths']
      # del features['rawlabelspaths']
      del features['imageids']

    return (features, labels)

  with tf.name_scope('input_pipeline'):
    dataset = prebatch_dataset(config, params)
    dataset = dataset.batch(get_temp_Nb(config, params.Nb))
    dataset = postbatch_dataset(dataset, config, params)
    dataset = dataset.map(_grouping, num_parallel_calls=NUM_PARALLEL_CALLS)
    dataset = dataset.prefetch(None)

  return dataset

# def _evaluate_preprocess(image, label, params):

#   _SIZE_FEATURE_EXTRACTOR = (params.height_feature_extractor, params.width_feature_extractor)

#   ## prepare
#   image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#   evaluation_lids2cids = _replacevoids(params.evaluation_problem_def['lids2cids'])
#   label = tf.gather(tf.cast(evaluation_lids2cids, tf.int32), tf.to_int32(label))

#   ## preprocess
#   proimage = tf.image.resize_images(image, _SIZE_FEATURE_EXTRACTOR)
#   prolabel = tf.image.resize_images(label[..., tf.newaxis],
#                                     _SIZE_FEATURE_EXTRACTOR,
#                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

#   proimage = from_0_1_to_m1_1(proimage)

#   print('debug: proimage, prolabel', proimage, prolabel)

#   return image, label, proimage, prolabel

# def _evaluate_parse_and_preprocess(im_la_files, data_location, params):
#   image, label, im_path, la_path = _load_and_decode(data_location, im_la_files)
#   image, label, proimage, prolabel = _evaluate_preprocess(image, label, params)

#   return image, label, proimage, prolabel, im_path, la_path

# def evaluate_input(config, params):

#   del config

#   data_location = params.dataset_directory
#   filenames_list = params.filelist_filepath
#   filenames_string = tf.cast(filenames_list, tf.string)

#   dataset = tf.data.TextLineDataset(filenames=filenames_string)

#   dataset = dataset.map(
#       functools.partial(_evaluate_parse_and_preprocess, data_location=data_location, params=params),
#       num_parallel_calls=30)
#   # IMPORTANT: if Nb > 1, then shape of dataset elements must be the same
#   dataset = dataset.batch(params.Nb)

#   def _grouping(rim, rla, pim, pla, imp, lap):
#     # group dataset elements as required by estimator
#     features = {
#         'rawimages': rim,
#         'proimages': pim,
#         'rawimagespaths': imp,
#         'rawlabelspaths': lap,
#         }
#     labels = {
#         'rawlabels': rla,
#         'prolabels': pla,
#         }

#     return (features, labels)

#   dataset = dataset.map(_grouping, num_parallel_calls=30)

#   return dataset

# def _predict_image_generator(params):
#   SUPPORTED_EXTENSIONS = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'ppm', 'PPM']
#   fnames = []
#   for se in SUPPORTED_EXTENSIONS:
#     fnames.extend(glob.glob(join(params.predict_dir, '*.' + se), recursive=True))

#   for im_fname in fnames:
#     start = datetime.now()
#     im = Image.open(im_fname)
#     # next line is time consuming (can take up to 400ms for im of 2 MPixels)
#     im_array = np.array(im)
#     # print('reading time:', datetime.now() - start)
#     yield im_array, im_fname.encode('utf-8'), im_array.shape[0], im_array.shape[1]

# def _predict_preprocess(image, params):
#   image.set_shape((None, None, 3))
#   image = tf.image.convert_image_dtype(image, tf.float32)
#   image = tf.image.resize_images(image, [params.height_feature_extractor, params.width_feature_extractor])

#   proimage = from_0_1_to_m1_1(image)

#   return proimage

# def predict_input(config, params):

#   del config

#   dataset = tf.data.Dataset.from_generator(lambda: _predict_image_generator(params),
#                                            output_types=(tf.uint8, tf.string, tf.int32, tf.int32))
#   dataset = dataset.map(lambda im, im_path, height, width: (
#       im_path, im, _predict_preprocess(im, params)), num_parallel_calls=30)
#   dataset = dataset.batch(params.Nb)
#   dataset = dataset.prefetch(None)

#   def _grouping(imp, rim, pim):
#     # group dataset elements as required by estimator
#     features = {'rawimages': rim,
#                 'proimages': pim,
#                 'rawimagespaths': imp}

#     return features

#   dataset = dataset.map(_grouping, num_parallel_calls=30)

#   return dataset

# def add_train_input_pipeline_arguments(argparser, ctx=None):
#   """
#   Add arguments required by the input pipeline.

#   Arguments:
#     argparser: an argparse.ArgumentParser object to add arguments
#     ctx: a context object with a suffix_name attribute
#   """

#   context_name = ctx.suffix_name if ctx else ''
#   argparser.add_argument('dataset_directory' + ('_' + context_name if ctx else ''), type=str,
#                          default='/media/panos/data/datasets/apolloscape/',
#                          help='Dataset directory including final /.')
#   argparser.add_argument('filelist_filepath' + ('_' + context_name if ctx else ''), type=str,
#                          default='/media/panos/data/datasets/apolloscape/public_image_lists/road01_ins_train.lst',
#                          help='List file as provided by the original dataset authors.')
#   argparser.add_argument('--preserve_aspect_ratio' + ('_' + context_name if ctx else ''), action='store_true',
#                          help='Resizes the input images respecting the aspect ratio using cropings.')

# def add_evaluate_input_pipeline_arguments(argparser):
#   """
#   Add arguments required by the input pipeline.

#   Arguments:
#     argparser: an argparse.ArgumentParser object to add arguments
#   """

#   argparser.add_argument('dataset_directory', type=str, default='/media/panos/data/datasets/apolloscape/',
#                          help='Dataset directory including final /.')
#   argparser.add_argument('filelist_filepath', type=str, default='/media/panos/data/datasets/apolloscape/public_image_lists/road01_ins_val.lst',
#                          help='List file as provided by the original dataset authors.')
