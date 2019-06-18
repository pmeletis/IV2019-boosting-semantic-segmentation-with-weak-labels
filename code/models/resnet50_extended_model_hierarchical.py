"""User-defined dense semantic segmentation models.
"""

import contextlib
import pprint
import functools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from models.resnet50_extended_feature_extractor import feature_extractor
from utils.utils import almost_equal
from utils.cross_replica_batch_normalization import cross_replica_batch_normalization
from input_pipelines.utils import get_temp_Nb

def model(mode, features, labels, config, params):
  """
  Arguments:
    features: tf.float32, Nb x hf x wf x Nc, (?, ?, ?, ?)

  Effective Receptive Field: ~200x200 pixels
  Return:
    predictions: a dict containing predictions:
      logits: tf.float32, Nb x hf x wf x Nc, (?, hf, wf, Nc)
      probabilities: tf.float32, Nb x hf x wf x Nc, (?, hf, wf, Nc)
      decisions: tf.int32, Nb x hf x wf, (?, hf, wf, Nc)
  """

  _validate_params(params)

  # ResNet-50 extended model require predefined channel (C) dimensions
  # set channel shape to 3 (RGB colors)
  features.set_shape((None, None, None, 3))
  # if group norm then batch need to be defined
  if params.norm_layer == 'group':
    features.set_shape((get_temp_Nb(config, params.Nb), None, None, None))

  # define arguments scope
  network_scope_args = {
      'norm_train_variables': params.norm_train_variables,
      'batch_norm_accumulate_statistics': params.batch_norm_accumulate_statistics,
      'norm_type': params.norm_layer}
  if mode == tf.estimator.ModeKeys.TRAIN:
    network_scope_args.update(
        weight_decay=params.regularization_weight,
        batch_norm_decay=params.batch_norm_decay,
        cross_replica_norm=params.cross_replica_norm)
  args_context = functools.partial(module_arg_scope, **network_scope_args)

  # build the feature extractor
  with tf.variable_scope('feature_extractor'), slim.arg_scope(args_context()):
    features, end_points = feature_extractor(mode, features, labels, config, params)
    # add optionally a PSP module
    if params.psp_module:
      with tf.variable_scope('pyramid_module'):
        features = _create_psp_module(features, params)

  def _bottleneck(features, scope):
    return resnet_v1.bottleneck(features,
                                features.shape[-1].value,
                                features.shape[-1].value,
                                1,
                                scope=scope)

  with tf.variable_scope('adaptation_module'), slim.arg_scope(args_context()):
    # Vistas: l1 features for classifying into 50(Vistas) + 1(vehicle) + 1(human) + 1(void) classes
    #         l2 features for classifying vehicle into 11(types of vehicle) + 1(void) classes
    #         l2 features for classifying human into 4(types of human) + 1(void) classes
    l1_features = _bottleneck(features, 'l1_features')
    l2_vehicle_features = _bottleneck(features, 'l2_vehicle_features')
    l2_human_features = _bottleneck(features, 'l2_human_features')

  ## create head: logits, probabilities and top-1 decisions
  ##   First the logits are created and then upsampled for memory efficiency.
  # if group normalization then groups must be less than channels -> 
  #   layer norm (1) works better than instance norm (output_Nclasses) for same hyperparameters
  with tf.variable_scope('softmax_classifier'), slim.arg_scope(args_context(groups=1)):
    def _conv2d(features, n_out, sc):
      return slim.conv2d(features, num_outputs=n_out, kernel_size=1, activation_fn=None, scope=sc)
    l1_logits = _conv2d(l1_features, 53 if params.per_pixel_dataset_name=='vistas' else 14, 'l1_logits')
    l2_vehicle_logits = _conv2d(l2_vehicle_features, 12 if params.per_pixel_dataset_name=='vistas' else 7, 'l2_vehicle_logits')
    l2_human_logits = _conv2d(l2_human_features, 5 if params.per_pixel_dataset_name=='vistas' else 3, 'l2_human_logits')
    l1_logits = _create_upsampler(l1_logits, params)
    l2_vehicle_logits = _create_upsampler(l2_vehicle_logits, params)
    l2_human_logits = _create_upsampler(l2_human_logits, params)

    l1_probs = tf.nn.softmax(l1_logits, name='l1_probabilities')
    l1_decs = tf.cast(tf.argmax(l1_probs, 3), tf.int32, name='l1_decisions')
    l2_vehicle_probs = tf.nn.softmax(l2_vehicle_logits, name='l2_vehicle_probabilities')
    l2_vehicle_decs = tf.cast(tf.argmax(l2_vehicle_probs, 3), tf.int32, name='l2_vehicle_decisions')
    l2_human_probs = tf.nn.softmax(l2_human_logits, name='l2_human_probabilities')
    l2_human_decs = tf.cast(tf.argmax(l2_human_probs, 3), tf.int32, name='l2_human_decisions')
    # generate final decisions
    if params.per_pixel_dataset_name == 'vistas':
      # human: 19->19, vehicle: 49->52
      l1_cids2common_cids = tf.cast([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
        63, 64, 65], tf.int32)
      l2_vehicle_cids2common_cids = tf.cast([52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65], tf.int32)
      l2_human_cids2common_cids = tf.cast([19, 20, 21, 22, 65], tf.int32)
    elif params.per_pixel_dataset_name == 'cityscapes':
      l1_cids2common_cids = tf.cast([
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10,11,13,19], tf.int32)
      l2_vehicle_cids2common_cids = tf.cast([13, 14, 15, 16, 17, 18, 19], tf.int32)
      l2_human_cids2common_cids = tf.cast([11, 12, 19], tf.int32)

    decs = tf.where(tf.equal(l1_decs, 49 if params.per_pixel_dataset_name=='vistas' else 12),
                    tf.gather(l2_vehicle_cids2common_cids, l2_vehicle_decs),
                    tf.where(tf.equal(l1_decs, 19 if params.per_pixel_dataset_name=='vistas' else 11),
                             tf.gather(l2_human_cids2common_cids, l2_human_decs),
                             tf.gather(l1_cids2common_cids, l1_decs)))

  ## model outputs groupped as predictions of the Estimator
  # WARNING: 'decisions' key is used internally so it must exist for now..
  predictions = {'l1_logits': l1_logits,
                 'l1_probabilities': l1_probs,
                 'l1_decisions': l1_decs,
                 'l2_vehicle_logits': l2_vehicle_logits,
                 'l2_vehicle_probabilities': l2_vehicle_probs,
                 'l2_vehicle_decisions': l2_vehicle_decs,
                 'l2_human_logits': l2_human_logits,
                 'l2_human_probabilities': l2_human_probs,
                 'l2_human_decisions': l2_human_decs,
                 'decisions': decs}

  # distribute is not yet supported in evaluate and predict
  if hasattr(params, 'distribute') and params.distribute:
    tower_context = tf.contrib.distribute.get_tower_context()
    assert tower_context
    twr_str = f"Tower {tower_context.tower_id}"
  else:
    twr_str = ''
  tf.logging.info(twr_str + " predictions:\n" + pprint.pformat(predictions, width=10))

  return features, end_points, predictions

def _create_upsampler(bottom, params):
  # upsample bottom depthwise to reach feature extractor output dimensions
  # bottom: Nb x hf//sfe x hf//sfe x C
  # upsampled: Nb x hf x hf x C
  # TODO: Does upsampler needs regularization??
  # TODO: align_corners=params.enable_xla: XLA implements only
  # align_corners=True for now, change it when XLA implements all

  C = bottom.shape[-1]
  spat_dims = np.array(bottom.shape.as_list()[1:3])
  hf, wf = params.height_feature_extractor, params.width_feature_extractor
  # WARNING: Resized images will be distorted if their original aspect ratio is
  # not the same as size (only in the case of bilinear resizing)
  if params.upsampling_method != 'no' and bottom.shape[1:3].is_fully_defined():
    assert almost_equal(
        spat_dims[0]/spat_dims[1],
        hf/wf,
        10**-1), (
            f"Resized images will be distorted if their original aspect ratio is "
            f"not the same as size: {spat_dims[0],spat_dims[1]}, {hf,wf}.")
  with tf.variable_scope('upsampling'):
    if params.upsampling_method == 'no':
      upsampled = bottom
    elif params.upsampling_method == 'bilinear':
      upsampled = tf.image.resize_images(bottom, [hf, wf], align_corners=True)
    elif params.upsampling_method == 'hybrid':
      # composite1: deconv upsample twice and then resize
      assert params.stride_feature_extractor in (4, 8, 16, 32), (
          'stride_feature_extractor must be 4, 8, 16 or 32.')
      upsampled = slim.conv2d_transpose(
          inputs=bottom,
          num_outputs=C,
          kernel_size=3,
          padding='SAME',
          activation_fn=None,
          weights_initializer=slim.variance_scaling_initializer(),
          weights_regularizer=slim.l2_regularizer(params.regularization_weight))
      upsampled = tf.image.resize_images(upsampled, [hf, wf], align_corners=True)
    else:
      raise ValueError('No such upsampling method.')

  return upsampled

def _create_psp_module(bottom, params):
  # Pyramid Scene Parsing
  hf, wf = params.height_feature_extractor, params.width_feature_extractor
  spatial_dimensions = np.array([hf, wf]) // params.stride_feature_extractor

  pool1 = slim.layers.avg_pool2d(bottom, spatial_dimensions, stride=spatial_dimensions)
  conv1 = slim.conv2d(pool1, params.feature_dims_decreased, 1)
  ups1 = tf.image.resize_images(conv1, tf.shape(bottom)[1:3], align_corners=True)
  pool2 = slim.layers.avg_pool2d(bottom, spatial_dimensions//2, stride=spatial_dimensions//2)
  conv2 = slim.conv2d(pool2, params.feature_dims_decreased, 1)
  ups2 = tf.image.resize_images(conv2, tf.shape(bottom)[1:3], align_corners=True)
  pool3 = slim.layers.avg_pool2d(bottom, spatial_dimensions//3, stride=spatial_dimensions//3)
  conv3 = slim.conv2d(pool3, params.feature_dims_decreased, 1)
  ups3 = tf.image.resize_images(conv3, tf.shape(bottom)[1:3], align_corners=True)
  pool6 = slim.layers.avg_pool2d(bottom, spatial_dimensions//6, stride=spatial_dimensions//6)
  conv6 = slim.conv2d(pool6, params.feature_dims_decreased, 1)
  ups6 = tf.image.resize_images(conv6, tf.shape(bottom)[1:3], align_corners=True)
  concated = tf.concat([bottom, ups1, ups2, ups3, ups6], 3)

  conv_final = slim.conv2d(concated, params.feature_dims_decreased, 1)

  return conv_final

# def _create_aspp_module(bottom, params):
#   # Atrous Spatial Pyramid Pooling (according to Rethinking Atrous Convolutions for SS)
#   # make rank is 4 and spatial dimensions are fully defined
#   bottom.shape.assert_has_rank(4)
#   bottom.shape[1:3].assert_is_fully_defined()
#   spatial_dimensions = bottom.shape.as_list()[1:3]

#   pool1 = slim.layers.avg_pool2d(bottom, spatial_dimensions, stride=spatial_dimensions)
#   conv1 = slim.conv2d(pool1, params.feature_dims_decreased, 1)
#   ups1 = tf.image.resize_images(conv1, spatial_dimensions, align_corners=True)
#   conv1 = slim.conv2d(bottom, params.feature_dims_decreased, 1, rate=1)
#   conv6 = slim.conv2d(bottom, params.feature_dims_decreased, 3, rate=6)
#   conv12 = slim.conv2d(bottom, params.feature_dims_decreased, 3, rate=12)
#   conv18 = slim.conv2d(bottom, params.feature_dims_decreased, 3, rate=18)
#   concated = tf.concat([ups1, conv1, conv6, conv12, conv18], 3)
#   conv_final = slim.conv2d(concated, params.feature_dims_decreased, 1)

#   return conv_final

def add_model_arguments(argparser):
  """
  Add arguments required by the model.

  Arguments:
    argparser: an argparse.ArgumentParser object to add arguments
  """

  argparser.add_argument('--stride_feature_extractor', type=int, default=8,
                         help='Output stride of the feature extractor. For the resnet_v1_* familly must be in {4,8,16,...}.')
  # TODO(panos): differentiate resnet_v1_101 model
  argparser.add_argument('--name_feature_extractor', type=str, default='resnet_v1_50',
                         choices=['resnet_v1_50', 'resnet_v1_101'], help='Feature extractor network.')

  # 1024 -> ||                                   ALGORITHM                                 || -> 1024  ::  h=512, s=512/512=1
  # 1024 -> || 1024 -> ||                     LEARNABLE NETWORK                 || -> 1024 || -> 1024  ::  hl=512, snet=512/512=1
  # 1024 -> || 1024 -> || 512 -> FEATURE EXTRACTOR -> 128 -> [UPSAMPLER -> 512] || -> 1024 || -> 1024  ::  hf=512, sfe=512/128=4
  argparser.add_argument('--feature_dims_decreased', type=int, default=256,
                         help='If >0 decreases feature dimensions of the feature extractor\'s output (usually 2048) to feature_dims_decreased using another convolutional layer.')
  argparser.add_argument('--fov_expansion_kernel_size', type=int, default=0,
                         help='If >0 increases the Field of View of the feature representation using an extra convolutional layer with this kernel size.')
  argparser.add_argument('--fov_expansion_kernel_rate', type=int, default=0,
                         help='If >0 increases the Field of View of the feature representation using an extra convolutional layer with this dilation rate.')
  argparser.add_argument('--upsampling_method', type=str, default='bilinear', choices=['no', 'bilinear', 'hybrid'],
                         help='No, Bilinear or hybrid upsampling are currently supported.')
  argparser.add_argument('--psp_module', action='store_true',
                         help='Whether to add Pyramid Scene Parsing module.')
  argparser.add_argument('--norm_layer', type=str, default='batch', choices=['batch', 'group'],
                         help='Select which type of normalization will be applied after ')
  argparser.add_argument('--cross_replica_norm', action='store_true',
                         help='During distributed training, normalization layer acts in cross replica context.')
  argparser.add_argument('--norm_train_variables', action='store_true',
                         help='Whether to add normalization layer variables (beta, gamma) into '
                              'trainable variables collection. Not providing this flag can switch off '
                              'training of normalization layer\'s variables.')
  argparser.add_argument('--batch_norm_accumulate_statistics', action='store_true',
                         help='Whether to accumulate moving mean and variances for batch norm '
                              'layers. Corresponds to `is_training` or `training` previous semantics, so the '
                              'normal behavior is to be provided during training and not during '
                              'inference. But any other combination may be also useful.')
  argparser.add_argument('--batch_norm_decay', type=float, default=0.9,
                         help='Decay rate of batch norm layer (decrease for smaller batch (Nb or image dims)).')


def _validate_params(params):
  # XOR(if one is greater than zero)
  if bool(params.fov_expansion_kernel_rate) != bool(params.fov_expansion_kernel_size):
    raise ValueError(f"One of params.{{fov_expansion_kernel_rate, fov_expansion_kernel_size}} "
                     "is set. In order to take effect both should be set.")

def module_arg_scope(
    # network params
    # variables_name_or_scope=None, argument scope cannot contain variable scopes
    weight_decay=0.0001,
    # norm related
    norm_type='batch',
    norm_epsilon=1e-5,
    norm_scale=True,
    norm_train_variables=True,
    cross_replica_norm=False,
    # batch norm related
    batch_norm_decay=0.997,
    batch_norm_accumulate_statistics=True,
    # group norm related
    groups=32,
    ):

  if norm_type not in ['batch', 'group']:
    raise ValueError('norm_type not valid.')

  batch_norm_params = {
      # tf.contrib.layers uses decay parameter
      'decay': batch_norm_decay,
      # tf.layers uses momentum parameter
      # 'momentum': batch_norm_decay,
      'epsilon': norm_epsilon,
      'scale': norm_scale,
      'trainable': norm_train_variables,
      # tf.contrib.layers uses is_training parameter
      'is_training': batch_norm_accumulate_statistics,
      # tf.layers uses training parameter
      # 'training': batch_norm_accumulate_statistics,
      # in TF r1.12 by default updates_collections is tf.GraphKeys.UPDATE_OPS
      # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
      }

  group_norm_params = {
      'groups': groups,
      'epsilon': norm_epsilon,
      'scale': norm_scale,
      'trainable': norm_train_variables}

  if norm_type == 'batch':
    # since TF r1.12 slim is less supported
    # slim uses tf.contrib.layers,
    # which is different from tf.layers (but uses tf.keras.layers inside),
    # which is different from tf.keras.layers
    normalizer_fn = tf.contrib.layers.batch_norm # tf.layers.batch_norm
    normalizer_params = batch_norm_params
    if cross_replica_norm:
      normalizer_fn = cross_replica_batch_normalization
  else:
    normalizer_fn = tf.contrib.layers.group_norm
    normalizer_params = group_norm_params
    if cross_replica_norm:
      raise ValueError('cross_replica_norm is supported only for batch normalization for now.')

  conv2d_params = {
      'weights_regularizer': slim.l2_regularizer(weight_decay),
      'weights_initializer': slim.variance_scaling_initializer(),
      'activation_fn': tf.nn.relu,
      'normalizer_fn': normalizer_fn,
      'normalizer_params': normalizer_params}

  with tf.contrib.framework.arg_scope(
      [slim.conv2d],
      **conv2d_params):
    with tf.contrib.framework.arg_scope(
        [slim.batch_norm],
        **batch_norm_params):
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.group_norm],
          **group_norm_params):
        with tf.contrib.framework.arg_scope(
            [slim.max_pool2d],
            padding='SAME') as arg_sc:
          return arg_sc
