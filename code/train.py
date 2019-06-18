"""Example of how to use Semantic Segmentation system for training.
"""

import sys
import collections
import os

import tensorflow as tf

from system_factory import SemanticSegmentation

from input_pipelines.heterogeneous_supervision.per_pixel_per_bbox_per_image import (
    train_input as train_fn, add_train_input_pipeline_arguments)

from models.resnet50_extended_model_hierarchical import model as model_fn, add_model_arguments

from utils.utils import SemanticSegmentationArguments
from utils.util_zip import zipit
from os.path import split, join, realpath

PATH_VISTAS_TFRECORDS = 'tfrecords/train.tfrecord'
PATH_CITYSCAPES_TFRECORDS = 'tfrecords/trainFine_v5.tfrecords'

def main(argv):
  ssargs = SemanticSegmentationArguments(mode=tf.estimator.ModeKeys.TRAIN)
  add_train_input_pipeline_arguments(ssargs.argparser)
  add_model_arguments(ssargs.argparser)
  # parse given arguments, add extra ones and validate
  settings = ssargs.parse_args(argv)
  if not os.path.exists(settings.init_ckpt_path):
    raise NotImplementedError(
        'For training Imagenet pretrained weights path must be hard-coded in utils.py/add_train_arguments')
  _add_extra_args(settings)

  system = SemanticSegmentation({'train': train_fn}, model_fn, settings)

  # save an instance of the code in logging directory at the moment of running for future reference
  zipit(split(realpath(__file__))[0], join(system.settings.log_dir, 'all_code.zip'))

  system.train()

def _add_extra_args(settings):
  # turn on training normalization layer variables and
  #   statistics accumulation for batch norm
  settings.norm_train_variables = True
  settings.batch_norm_accumulate_statistics = True

  if settings.per_pixel_dataset_name == 'vistas':
    settings.Ntrain = 18000
    settings.training_problem_def_path = 'problem_definitions/vistas/problem01.json'
    settings.tfrecords_path_per_pixel = PATH_VISTAS_TFRECORDS
    # for training with Nb=4 use 590, 814
    settings.height_feature_extractor = 621
    settings.width_feature_extractor = 855
  elif settings.per_pixel_dataset_name == 'cityscapes':
    settings.Ntrain = 2975
    settings.training_problem_def_path = 'problem_definitions/cityscapes/problem01.json'
    settings.tfrecords_path_per_pixel = PATH_CITYSCAPES_TFRECORDS
    settings.height_feature_extractor = 512
    settings.width_feature_extractor = 1024

  settings.Nb_per_pixel = 4
  settings.Nb_per_bbox = 8
  settings.Nb_per_image = 4
  settings.Nb = settings.Nb_per_pixel
  settings.preserve_aspect_ratio_per_pixel = False
  settings.preserve_aspect_ratio_per_bbox = True
  settings.preserve_aspect_ratio_per_image = True

if __name__ == '__main__':
  if not os.path.exists(PATH_CITYSCAPES_TFRECORDS) or not os.path.exists(PATH_VISTAS_TFRECORDS):
    raise NotImplementedError('For training Cityscapes and Vistas tfrecords path must be hard-coded in train.py')
  main(sys.argv[1:])
