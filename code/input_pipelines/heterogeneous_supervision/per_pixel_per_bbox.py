"""
Training input function:
  dense per_pixel supervision (per-pixel annotations) +
  sparse OpenImages supervision (per bbox and per image annotations)
"""

import sys
import os.path as op
import functools
import copy
sys.path.append(op.split(op.split(op.realpath(__file__))[0])[0])

import tensorflow as tf

from input_pipelines.open_images.input_subset_bboxes_v2 import (
    train_input as train_input_per_bbox)

def train_input(config, params):
  """
  Returns a tf.data.Dataset for training from Open Images and per_pixel datasets
  """

  if params.per_pixel_dataset_name == 'vistas':
    from input_pipelines.vistas.input_vistas import (
        train_input as train_input_per_pixel,
        add_train_input_pipeline_arguments as add_train_args_per_pixel)
  elif params.per_pixel_dataset_name == 'cityscapes':
    from input_pipelines.cityscapes.input_cityscapes import (
        train_input as train_input_per_pixel,
        add_train_input_pipeline_arguments as add_train_args_per_pixel)

  params_per_pixel = copy.deepcopy(params)
  params_per_pixel.Nb = params.Nb_per_pixel
  params_per_pixel.tfrecords_path = params.tfrecords_path_per_pixel
  params_per_pixel.preserve_aspect_ratio = params.preserve_aspect_ratio_per_pixel
  per_pixel_dataset = train_input_per_pixel(config, params_per_pixel)

  params_per_bbox = copy.deepcopy(params)
  params_per_bbox.Nb = params.Nb_per_bbox
  params_per_bbox.preserve_aspect_ratio = params.preserve_aspect_ratio_per_bbox
  per_bbox_dataset = train_input_per_bbox(config, params_per_bbox)

  def _concat_datasets(per_pixel, per_bbox):
    # (({proimages: (?, 512, 512, ?), rawimagespaths: (?,), rawlabelspaths: (?,)},
    #   {prolabels: (?, 512, 512)}),
    #  ({proimages: (?, 512, 512, ?), imageids: (?,)},
    #   {prolabels: (?, 512, 512, ?)}))

    fe_vi, la_vi = per_pixel
    fe_oi, la_oi = per_bbox

    features = {
        'proimages': tf.concat([fe_vi['proimages'], fe_oi['proimages']], 0),
    }

    # due to bug in TF
    if not params.distribute:
      features.update({
          'imageids': fe_oi['imageids'],
          'rawimagespaths': fe_vi['rawimagespaths'],
          'rawlabelspaths': fe_vi['rawlabelspaths']})

    labels = {
        'prolabels_per_pixel': la_vi['prolabels'],
        'prolabels_per_bbox': la_oi['prolabels'],
    }

    return features, labels

  with tf.name_scope('combined_input_pipeline'):
    dataset = tf.data.Dataset.zip((per_pixel_dataset, per_bbox_dataset))
    dataset = dataset.map(_concat_datasets, num_parallel_calls=15)
    # print(dataset)
    # dataset = dataset.prefetch(None)

  return dataset

def add_train_input_pipeline_arguments(argparser):
  """
  Add arguments required by the input pipeline.

  Arguments:
    argparser: an argparse.ArgumentParser object to add arguments
  """
  # class per_pixel_ctx(object):
  #   suffix_name = 'per_pixel'

  # class per_bbox_ctx(object):
  #   suffix_name = 'per_bbox'

  # add_train_args_per_pixel(argparser, per_pixel_ctx)
  # add_train_args_per_bbox(argparser, per_bbox_ctx)
  pass
