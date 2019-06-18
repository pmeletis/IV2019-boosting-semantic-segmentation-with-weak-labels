"""Input pipelines utilities, common code."""

import functools

import tensorflow as tf
from tensorflow.python.util.deprecation import deprecated
# from tensorflow.python.eager import context
from tensorflow.image import ResizeMethod as Method

from utils.utils import resize_images_or_labels, tensor_shape

@deprecated('2018-07-09',
            'Please use resize_images_or_labels from utils/utils.py.')
def adjust_images_labels_size(images, labels, size):
  """
  Smartly adjust `images` and `labels` spatial dimensions to `size`.
  The input of an FCN feature extractor for semantic segmentation must normally be of
    fixed spatial dimensions. This function adjusts the spatial dimensions of input images
    and corresponding labels to a fixed predefined size. The "smart adjustment" is done
    according to the following rules. First the input is resized, preserving the aspect
    ratio, to the smallest size, which is at least bigger than `size` and then cropped to `size`.
  There are 9 possibilities for comparing the dimensions of dataset images (H, W)
  and feature extractor (h, w). They stem from permutations with repetition (n_\Pi_r = n^r)
  of n = 3 objects (<, >, =) taken by r = 2 (h ? H, w ? W).
      H ? h   W ? w                     H ? h   W ? w
        =       =    random crop          =       <    upscale and random crop
        =       >    random crop          <       =    upscale and random crop
        >       =    random crop          >       <    upscale and random crop
                                          <       <    upscale and random crop
                                          <       >    upscale and random crop
                                          >       >    downscale and random crop
  These cases can be summarized to the following pseudocode,
    which creates the smallest possible D' >= d, where d = (h, w), D = (H, W):
      if reduce_any(D < d) or reduce_all(D > d):
        upscale to D' = D * reduce_max(d/D)
      random crop
  These transformations preserve the relative scale of objects across different image sizes and
    are "meaningful" for training a network in combination with adjusted sized inference.

  Arguments:
    images: tf.float32, (?, ?, ?, 3), in [0, 1)
    labels: tf.int32, (?, ?, ?)
    size: Python tuple, (2,), a valid spatial size in height, width order

  Return:
    images, labels: ...
  """

  assert isinstance(size, tuple), 'size must be a tuple.'
  # TODO(panos): add input checks

  shape = tf.shape(images)
  spatial_shape = shape[1:3]

  upscale_condition = tf.reduce_any(tf.less(spatial_shape, size))
  downscale_condition = tf.reduce_all(tf.greater(spatial_shape, size))
  factor = tf.cast(tf.reduce_max(size/spatial_shape), tf.float32)

  def _resize_images_labels(images, labels, size):
    images = tf.image.resize_images(images, size)
    labels = tf.image.resize_images(labels[..., tf.newaxis],
                                    size,
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
    return images, labels

  def _true_fn():
    return _resize_images_labels(
        images,
        labels,
        tf.cast(tf.ceil(factor*tf.cast(spatial_shape, tf.float32)), tf.int32))

  def _false_fn():
    return images, labels

  combined_condition = tf.logical_or(upscale_condition, downscale_condition)
  images, labels = tf.cond(combined_condition, _true_fn, _false_fn)

  def _random_crop(images, labels, size):
    # TODO(panos): check images and labels spatial size statically,
    #   if it is defined and is the same as size do not add random_crop ops
    # convert images to tf.int32 to concat and random crop the same area
    images = tf.cast(tf.image.convert_image_dtype(images, tf.uint8), tf.int32)
    concated = tf.concat([images, labels[..., tf.newaxis]], axis=3)
    crop_size = (images.shape[0].value,) + size + (4,)
    print('debug:concated,crop_size:', concated, crop_size)
    concated_cropped = tf.random_crop(concated, crop_size)
    # convert images back to tf.float32
    images = tf.image.convert_image_dtype(tf.cast(concated_cropped[..., :3], tf.uint8), tf.float32)
    labels = concated_cropped[..., 3]
    return images, labels

  images, labels = _random_crop(images, labels, size)

  return images, labels

def from_0_1_to_m1_1(images):
  """
  Center images from [0, 1) to [-1, 1).

  Arguments:
    images: tf.float32, in [0, 1), of any dimensions
  
  Return:
    images linearly scaled to [-1, 1)
  """

  # TODO(panos): generalize to any range
  # shifting from [0, 1) to [-1, 1) is equivalent to assuming 0.5 mean
  mean = 0.5
  proimages = (images - mean)/mean

  return proimages

# TODO(panos): as noted in MirrorStrategy, in a multi-gpu setting the effective
#   batch size is num_gpus * Nb, so deal with this until per batch broadcasting
#   is implemented in core tensorflow package
# by default all available gpus of the machine are used
def get_temp_Nb(runconfig, Nb):
  if runconfig.train_distribute:
    div, mod = divmod(Nb, runconfig.train_distribute.num_towers)
    assert not mod, 'for now Nb must be divisible by the number of available GPUs.'
    return div
  else:
    return Nb
    #assert False, 'Code should have not reached this point, contact maintainers.'

@deprecated('2018-08-25',
            'Please use resize_images_and_labels from input_pipelines/utils.py.')
def resize_images_and_labels_preserving_aspect_ratio_with_cropping(images, labels, target_size):
  """
  The main purpose of this function is to crop the corresponding patches for images and labels,
  with preserving the aspect ratio using random cropping. Assumes images and labels have the same
    spatial size, so cropping to the same size is possible.

  Args:
    images: tf.float32, (?, ?, ?, ?)
    labels: tf.int32, (?, ?, ?)
    refer to resize_images_or_labels for other arguments

  Returns:
    images and labels resized to target_size, i.e. statically defined spatial dimensions,
      preserving the aspect ratio, using random cropping if necessary
  """

  # work with dynamic shape
  assertions = [tf.assert_equal(tensor_shape(images, 4)[1:3],
                                tensor_shape(labels, 3)[1:3])]
  with tf.control_dependencies(assertions):
    images = tf.identity(images)

  # resize using mode='max' without cropping so that target_size fits
  #   tightly in images and labels
  _resize = functools.partial(resize_images_or_labels,
                              candidate_size=target_size,
                              preserve_aspect_ratio=True,
                              mode='max',
                              crop=None)
  proimages = _resize(features=images, method=Method.BILINEAR)
  prolabels = _resize(features=labels, method=Method.NEAREST_NEIGHBOR)

  # generate a random cropping offset
  proimages_extra_size = tf.subtract(tensor_shape(proimages, 4)[1:3], target_size)
  _rnd = functools.partial(tf.random_uniform, (), dtype=tf.int32)
  offset = tf.stack([
      _rnd(maxval=proimages_extra_size[0] + 1),
      _rnd(maxval=proimages_extra_size[1] + 1)])

  # crop
  begin = offset
  end = tf.add(begin, target_size)
  proimages = proimages[:, begin[0]:end[0], begin[1]:end[1], :]
  prolabels = prolabels[:, begin[0]:end[0], begin[1]:end[1]]

  # before slicing is defined by tensors so merge with known shape
  #   to define shape statically
  proimages.set_shape((None, *target_size, None))
  prolabels.set_shape((None, *target_size))

  return proimages, prolabels

def resize_images_and_labels(images, labels, target_size, preserve_aspect_ratio=False):
  """
  The main purpose of this function is to crop the corresponding patches for images and labels,
  if `preserve_aspect_ratio` is set using random cropping. Assumes images and labels have the same
    spatial size.

  Args:
    images: tf.float32, (?, ?, ?, ?)
    labels: tf.int32, (?, ?, ?) for one-hot (sparse) labels,
            or tf.float32, (?, ?, ?, ?) for multi-hot (dense) labels
    refer to resize_images_or_labels for other arguments

  Returns:
    images and labels resized to target_size, i.e. statically defined spatial dimensions
  """

  # work with dynamic shape
  rank_labels = labels.shape.ndims
  assert rank_labels, 'rank of labels must be statically defined.'
  assertions = [tf.assert_equal(tensor_shape(images, 4)[1:3],
                                tensor_shape(labels, labels.shape.ndims)[1:3])]
  with tf.control_dependencies(assertions):
    images = tf.identity(images)

  # resize using mode='max' without cropping so that target_size fits
  #   tightly in images and labels
  _resize = functools.partial(resize_images_or_labels,
                              candidate_size=target_size,
                              preserve_aspect_ratio=preserve_aspect_ratio,
                              mode='max',
                              crop=None)
  proimages = _resize(features=images, method=Method.BILINEAR)
  prolabels = _resize(features=labels, method=Method.NEAREST_NEIGHBOR)

  # only if preserve_aspect_ratio is set the output of resize_images_or_labels is
  #   not guaranteed to have target_size dimensions
  if preserve_aspect_ratio:
    # generate a random cropping offset
    proimages_extra_size = tf.subtract(tensor_shape(proimages, 4)[1:3], target_size)
    _rnd = functools.partial(tf.random_uniform, (), dtype=tf.int32)
    offset = tf.stack([
        _rnd(maxval=proimages_extra_size[0] + 1),
        _rnd(maxval=proimages_extra_size[1] + 1)])

    # crop
    begin = offset
    end = tf.add(begin, target_size)
    proimages = proimages[:, begin[0]:end[0], begin[1]:end[1], :]
    if rank_labels == 3:
      prolabels = prolabels[:, begin[0]:end[0], begin[1]:end[1]]
    elif rank_labels == 4:
      prolabels = prolabels[:, begin[0]:end[0], begin[1]:end[1], :]

    # above slicing is defined by tensors so merge with known shape
    #   to define shape statically
    proimages.set_shape((None, *target_size, None))
    if rank_labels == 3:
      prolabels.set_shape((None, *target_size))
    elif rank_labels == 4:
      prolabels.set_shape((None, *target_size, None))
    

  # sanity checks
  proimages.shape[1:3].assert_is_fully_defined()
  prolabels.shape[1:3].assert_is_fully_defined()

  return proimages, prolabels
