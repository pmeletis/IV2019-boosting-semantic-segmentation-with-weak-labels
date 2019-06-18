"""User-defined feature extractor for dense semantic segmentation model.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

def feature_extractor(mode, features, labels, config, params):
  """Fully Convolutional ResNet-50 type feature extractor for Semantic Segmentation.

  This function returns a feature extractor.
  First, the base feature extractor is created, which consists of a
  predefined network that is adapted for the problem of SS.
  Then, an optional extension to the feature extractor is created
  (in series with the base) to deal the with feature dimensions and
  the receptive field of the feature representation specialized to SS.
  """

  del mode, labels, config

  # build base of feature extractor
  with tf.variable_scope('base'):
    # if num_classes=None no logits layer is created
    # if global_pool=False model is used for dense output
    fe, end_points = resnet_v1.resnet_v1_50(
        features,
        num_classes=None,
        is_training=params.batch_norm_accumulate_statistics,
        global_pool=False,
        output_stride=params.stride_feature_extractor)

  # build extension to feature extractor, which decreases feature dimensions
  #   and increase field of view of feature extractor in a memory and 
  #   computational efficient way
  # TODO: add to end_points the outputs of next layers
  with tf.variable_scope('extension'):
    # WARNING: this scope assumes that slim.conv2d uses slim.batch_norm
    #   for the batch normalization, which holds at least up to TF v1.4
    if params.feature_dims_decreased > 0:
      fe = slim.conv2d(fe,
                       num_outputs=params.feature_dims_decreased,
                       kernel_size=1,
                       scope='decrease_fdims')
    if params.fov_expansion_kernel_rate > 0 and params.fov_expansion_kernel_size > 0:
      fe = slim.conv2d(fe,
                       num_outputs=fe.shape[-1],
                       kernel_size=params.fov_expansion_kernel_size,
                       rate=params.fov_expansion_kernel_rate,
                       scope='increase_fov')

  return fe, end_points
