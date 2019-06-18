"""
A library of preprocessing functions for semantic segmentation.
(c) Panagiotis Meletis, 2017-2018

Transformations implemented:
1) Geometric
  i)   random upscaling
  ii)  random downscaling
  iii) random scaling
  iv)  random flipping
2) Illumination
  v)   random color
  vi)  random blur
"""

import functools
import numpy as np
import cv2
import tensorflow as tf

def random_scaling(images, labels, poi, unlabeled_cid, enable_assertions=False):
  """
  Per image randomly applies upscaling or downscaling with 1/2 probability.
  For more info see random_upsaling and random_downscaling functions.
  images, labels: must have defined batch dimension (0) due to stacking and unstacking

  See random_upscaling and random_downscaling for more information and parameters.
  """

  upscaled_features = random_upscaling(images, labels, poi, enable_assertions)
  downscaled_features = random_downscaling(images, labels, poi, unlabeled_cid, enable_assertions)
  selectors = tf.greater(tf.random_uniform((images.shape[0].value,)), 0.5)
  proimages = tf.where(selectors, upscaled_features[0], downscaled_features[0])
  prolabels = tf.where(selectors, upscaled_features[1], downscaled_features[1])

  return proimages, prolabels

def random_upscaling(images, labels, poi, enable_assertions=False):
  """
  Applies per image a random upscaling. For each image, a random crop is upscaled
  to images initial spatial size.

  Arguments:
    images: 4D tf.Tensor (Nbx?x?x?), tf.float32, in [0,1)
    labels: 3D tf.Tensor (Nbx?x?), tf.int32
    poi: [low, high] range to randomly select upscaling factor from
    enable_assertions: enables arguments specification tests for verification,
      keep it false if you want less operations in the graph and faster execution

  Info:
    Nb: number of elements per batch
    no static shapes are required, except for batch dimension (0) of images and labels
      due to stacking and unstacking, all other shapes can be dynamically defined,
      only ranks have to be static
    images and labels must have the same batch and spatial size, i.e. dimensions 0, 1, 2
    assumes that each image corresponds to the respective label in the batch
    poi: low >= 1.0, low <= high

  Return:
    proimages: same shape, dtype, range as images
    prolabels: same shape, dtype, range as labels
  """

  # TODO(panos): enable poi per image
  # TODO(panos): enable single image (3D)

  # assertions
  # TODO(panos): name scope is not needed if assertions are disabled
  with tf.name_scope('random_upscaling_checks'):
    # classes
    assert isinstance(images, tf.Tensor), 'images must be a tf.Tensor.'
    assert isinstance(labels, tf.Tensor), 'labels must be a tf.Tensor.'
    assert isinstance(poi, (list, tuple)), 'poi must be a Python list or tuple.'
    # shapes
    assert len(poi) == 2, 'poi must have length 2.'
    images.shape.assert_has_rank(4)
    labels.shape.assert_has_rank(3)
    images_shape = tf.shape(images)
    labels_shape = tf.shape(labels)
    # check statically defined batch size
    # tf.Dimension __eq__ returns None if either shape is None or the comparison result
    # thus no need to check for not equal to tf.Dimension(None)
    assert images.shape[0] == labels.shape[0], (
        f"Batch size for images ({images.shape[0]}) and labels ({labels.shape[0]}) "
        "must be equal and defined (not None).")
    # check dynamically defined shape
    if enable_assertions:
      with tf.control_dependencies(
          [tf.assert_equal(images_shape[:3],
                           labels_shape,
                           message='images and labels must have same batch and spatial size.')]):
        images = tf.identity(images)
    # types
    assert all([isinstance(poi[0], float), isinstance(poi[1], float)]), (
        'poi must be a Python float sequence.')
    assert images.dtype == tf.float32, 'images must have tf.float32 tf.DType.'
    assert labels.dtype == tf.int32, 'labels must have tf.int32 tf.DType.'
    # values
    # 10.0 is a safe hard-coded value
    assert all([poi[0] >= 1.0, poi[1] <= 10.0, poi[0] <= poi[1]]), (
        'poi values must be in range [1.0, 10.0] and ordered.')
    if enable_assertions:
      with tf.control_dependencies(
          [tf.assert_greater_equal(images, tf.constant(0.0, dtype=tf.float32)),
           tf.assert_less(images, tf.constant(1.0, dtype=tf.float32))]):
        images = tf.identity(images)
      # with tf.control_dependencies(
      #     [tf.assert_greater_equal(label, tf.constant(0, dtype=tf.int32))]):
      #   label = tf.identity(label)

  # save computations
  if poi[0] == poi[1] == 1.0:
    return tf.identity(images), tf.identity(labels), tf.constant(1.0, dtype=tf.float32)

  #
  poi = [1/p for p in poi]
  poi = list(reversed(poi))

  # convert image to tf.int32 so it can be concatenated with label
  images = tf.cast(tf.image.convert_image_dtype(images, tf.uint8), tf.int32)
  concated = tf.concat([images, labels[..., tf.newaxis]], axis=3)

  # batch size (Nb) should be statically defined
  Nb = concated.get_shape().as_list()[0]
  concated_shape = tf.shape(concated)
  con_size, Nchannels = concated_shape[1:3], concated_shape[3]
  poi = tf.cast(poi, tf.float32)
  # random_factors.shape = (Nb, 1) so it is broadcastable to con_size
  random_factors = tf.random_uniform((Nb, 1), minval=poi[0], maxval=poi[1])
  # random_factors = tf.Print(random_factors, [random_factors], summarize=Nb)
  random_sizes = tf.cast(
      tf.floor(random_factors * tf.cast(con_size, dtype=tf.float32)),
      tf.int32)
  # random_crop_sizes.shape = (Nb, 3)
  random_crop_sizes = tf.concat([random_sizes, ((Nchannels,),)*Nb], axis=1)

  # unbatch concated
  cons = tf.unstack(concated)
  proimages = []
  prolabels = []
  # TODO(panos): convert to tf.while loop so the number of operations installed in
  #   the graph is independent of the Nb, i.e. times of next loop is executed
  for i, con in enumerate(cons):
    con_cropped = tf.random_crop(con, random_crop_sizes[i])
    # convert to tf.uint8 and then back to float32
    proimage = tf.image.convert_image_dtype(tf.cast(con_cropped[..., :3], tf.uint8), tf.float32)
    prolabel = con_cropped[..., 3]
    # reshape to initial shape
    proimages.append(tf.image.resize_images(proimage, con_size))
    prolabels.append(tf.image.resize_images(prolabel[..., tf.newaxis],
                                            con_size,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0])

  # rebatch
  proimages = tf.stack(proimages)
  prolabels = tf.stack(prolabels)

  return proimages, prolabels

def random_downscaling(images, labels, poi, unlabeled_cid, enable_assertions=False):
  """
  Applies per image a random downscaling. Each image is randomly downscaled and placed in a middle of a black image.
  Also, downscales the label and assigns pixels outside the downscaled label to unlabeled class.
  This operation creates a new unlabeled class in label, [more info to be added soon].

  Arguments:
    images: 4D tf.Tensor (Nbx?x?x?), tf.float32, in [0,1)
    labels: 3D tf.Tensor (Nbx?x?), tf.int32
    poi: [low, high] range to randomly select downscaling factor from
    unlabeled_cid: Python int, unlabeled class id, this augmentation introduces unlabeled pixels,
      and if this class does not exist an id should be provided and proper external handling
      should be implemented
    enable_assertions: enables arguments specification tests for verification,
      keep it false if you want less operations in the graph and faster execution

  Nb: number of elements per batch
  no static shapes are required, except for batch dimension (0) of images and labels
      due to stacking and unstacking, all other shapes can be dynamically defined,
      only ranks have to be static
  images and labels must have the same batch and spatial size, i.e. dimensions 0, 1, 2
  assumes that each image corresponds to the respective label in the batch
  poi: low >= 1.0, low <= high

  Return:
    proimages: same shape, dtype, range as images
    prolabels: same shape, dtype, range as labels
  """

  # assertions
  # TODO(panos): name scope is not needed if assertions are disabled
  with tf.name_scope('random_upscaling_checks'):
    # classes
    assert isinstance(images, tf.Tensor), 'images must be a tf.Tensor.'
    assert isinstance(labels, tf.Tensor), 'labels must be a tf.Tensor.'
    assert isinstance(poi, (list, tuple)), 'poi must be a Python list or tuple.'
    assert isinstance(unlabeled_cid, int), 'unlabeled_cid must be a Python int.'
    # shapes
    assert len(poi) == 2, 'poi must have length 2.'
    images.shape.assert_has_rank(4)
    labels.shape.assert_has_rank(3)
    images_shape = tf.shape(images)
    labels_shape = tf.shape(labels)
    # check statically defined batch size
    # tf.Dimension __eq__ returns None if either shape is None or the comparison result
    # thus no need to check for not equal to tf.Dimension(None)
    assert images.shape[0] == labels.shape[0], (
        f"Batch size for images ({images.shape[0]}) and labels ({labels.shape[0]}) "
        "must be equal and defined (not None).")
    # check dynamically defined shape
    if enable_assertions:
      with tf.control_dependencies(
          [tf.assert_equal(images_shape[:3],
                           labels_shape,
                           message='images and labels must have same batch and spatial size.')]):
        images = tf.identity(images)
    # types
    assert all([isinstance(poi[0], float), isinstance(poi[1], float)]), (
        'poi must be a Python float sequence.')
    assert images.dtype == tf.float32, 'images must have tf.float32 tf.DType.'
    assert labels.dtype == tf.int32, 'labels must have tf.int32 tf.DType.'
    # values
    # 10.0 is a safe hard-coded value
    assert all([poi[0] >= 1.0, poi[1] <= 10.0, poi[0] <= poi[1]]), (
        'poi values must be in range [1.0, 10.0] and ordered.')
    if enable_assertions:
      with tf.control_dependencies(
          [tf.assert_greater_equal(images, tf.constant(0.0, dtype=tf.float32)),
           tf.assert_less(images, tf.constant(1.0, dtype=tf.float32))]):
        images = tf.identity(images)
      # with tf.control_dependencies(
      #     [tf.assert_greater_equal(label, tf.constant(0, dtype=tf.int32))]):
      #   label = tf.identity(label)

  # save computations
  if poi[0] == poi[1] == 1.0:
    return tf.identity(images), tf.identity(labels), tf.constant(1.0, dtype=tf.float32)

  #
  poi = [1/p for p in poi]
  poi = list(reversed(poi))

  # batch size (Nb) should be statically defined
  Nb = images.get_shape().as_list()[0]
  images_shape = tf.shape(images)
  image_size = images_shape[1:3]
  poi = tf.cast(poi, tf.float32)
  # random_factors.shape = (Nb, 1) so it is broadcastable to con_size
  random_factors = tf.random_uniform((Nb, 1), minval=poi[0], maxval=poi[1])
  # random_factors = tf.Print(random_factors, [random_factors], summarize=Nb)
  random_sizes = tf.cast(
      tf.floor(random_factors * tf.cast(image_size, dtype=tf.float32)),
      tf.int32)

  # unbatch concated
  proimages = []
  prolabels = []
  # TODO(panos): convert to tf.while loop so the number of operations installed in
  #   the graph is independent of the Nb, i.e. times of next loop is executed
  for i, (im, la) in enumerate(zip(tf.unstack(images), tf.unstack(labels))):
    proim = tf.image.resize_images(im, random_sizes[i])
    prola = tf.image.resize_images(la[..., tf.newaxis],
                                   random_sizes[i],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

    size_to_pad = image_size - tf.shape(proim)[:2]
    paddings = tf.floordiv(size_to_pad, 2)
    paddings = tf.stack([paddings, size_to_pad - paddings], axis=1)

    # REFLECT mode may be better in models with batch norm, so as to affect as
    #   least as possible batch statistics
    # CONSTANT with reduce_mean does not change the mean of proim
    proim = tf.pad(proim,
                   tf.concat([paddings, [[0, 0]]], axis=0),
                  #  mode="REFLECT",
                   mode="CONSTANT",
                   constant_values=tf.reduce_mean(proim),
                  )

    prola = tf.pad(prola,
                   paddings,
                   mode="CONSTANT",
                   constant_values=tf.constant(unlabeled_cid, dtype=tf.int32))

    # proim and prola should be padded to initial shape
    # merge with known shape, since tf.pad outputs not defines shape due to tf.shape
    proim.set_shape(images.shape[1:])
    prola.set_shape(labels.shape[1:])

    proimages.append(proim)
    prolabels.append(prola)

  # rebatch
  proimages = tf.stack(proimages)
  prolabels = tf.stack(prolabels)

  return proimages, prolabels

def random_flipping(images, labels):
  """
  Applies per image random left or right flipping with probability 1/2.
  TODO(panos): add more info
  images, labels: must have defined batch dimension (0) due to stacking and unstacking

  IMPORTANT: Make sure that every class in the labels is equivariant to horizontal flipping,
    e.g. most of the traffic signs are not equivariant to horizontal flipping.
  """

  # TODO(panos): add assertion tests

  # convert image to tf.int32 so it can be concatenated with label
  images = tf.cast(tf.image.convert_image_dtype(images, tf.uint8), tf.int32)
  concated = tf.concat([images, labels[..., tf.newaxis]], axis=3)

  proconcated = tf.stack(
      [tf.image.random_flip_left_right(concat) for concat in tf.unstack(concated)])

  # convert to tf.uint8 and them back to float32
  proimages = tf.image.convert_image_dtype(tf.cast(proconcated[..., :3], tf.uint8), tf.float32)
  prolabels = proconcated[..., 3]

  return proimages, prolabels

def random_color(images):
  """
  Applies per image, a series of transformations (brightness, saturation, hue, contrast)
    or returns the original image with probability 1/2.

  Arguments:
    images: 4D tf.Tensor (NbxHxWxC), tf.float32, in [0,1)

  Nb: number of images per batch
  C: number of channels (e.g. 3 for RGB)
  images: must have defined batch dimension (0) due to stacking and unstacking

  Returns:
    proimages: same shape, dtype, range as images
  """

  # TODO(panos): add assertion tests

  col_r = tf.random_uniform([], minval=0, maxval=8, dtype=tf.int32)
  def maybe_transform(image):
    def _select_order(color_ordering):
      return functools.partial(distort_color, image, color_ordering=color_ordering)
    return tf.case({tf.equal(col_r, 0): _select_order(0),
                    tf.equal(col_r, 1): _select_order(1),
                    tf.equal(col_r, 2): _select_order(2),
                    tf.equal(col_r, 3): _select_order(3),
                    tf.equal(col_r, 4): lambda: image,
                    tf.equal(col_r, 5): lambda: image,
                    tf.equal(col_r, 6): lambda: image,
                    tf.equal(col_r, 7): lambda: image},
                   default=lambda: image)

  proimages = tf.stack([maybe_transform(image) for image in tf.unstack(images)])

  return proimages

def distort_color(image, color_ordering=0, scope=None):
  """Distort the color of a Tensor image.
  NOTE: distort_color function from tensorflow/models

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  brightness_max_delta = 32. / 255.
  lower = 0.7
  upper = 1.3
  hue_max_delta = 0.1
  with tf.name_scope(scope, 'distort_color', [image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
      image = tf.image.random_saturation(image, lower=lower, upper=upper)
      image = tf.image.random_hue(image, max_delta=hue_max_delta)
      image = tf.image.random_contrast(image, lower=lower, upper=upper)
    elif color_ordering == 1:
      image = tf.image.random_saturation(image, lower=lower, upper=upper)
      image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
      image = tf.image.random_contrast(image, lower=lower, upper=upper)
      image = tf.image.random_hue(image, max_delta=hue_max_delta)
    elif color_ordering == 2:
      image = tf.image.random_contrast(image, lower=lower, upper=upper)
      image = tf.image.random_hue(image, max_delta=hue_max_delta)
      image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
      image = tf.image.random_saturation(image, lower=lower, upper=upper)
    elif color_ordering == 3:
      image = tf.image.random_hue(image, max_delta=hue_max_delta)
      image = tf.image.random_saturation(image, lower=lower, upper=upper)
      image = tf.image.random_contrast(image, lower=lower, upper=upper)
      image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
    else:
      raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def random_blur(images):
  """
  Applies per image, blurring (median or bilateral) or returns the original image with
    probability 1/2.

  Arguments:
    images: 4D tf.Tensor (NbxHxWxC), tf.float32, in [0,1)

  Nb: number of images per batch
  C: number of channels (e.g. 3 for RGB)
  images: must have defined batch dimension (0) due to stacking and unstacking

  Returns:
    proimages: same shape, dtype, range as images
  """

  # TODO(panos): add assertion tests

  blu_r = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
  def maybe_transform(image):
    def _select_blur(blur_selector):
      return functools.partial(distort_blur, image, blur_selector=blur_selector)
    return tf.case({tf.equal(blu_r, 0): _select_blur(0),
                    tf.equal(blu_r, 1): _select_blur(1),
                    tf.equal(blu_r, 2): lambda: image,
                    tf.equal(blu_r, 3): lambda: image},
                   default=lambda: image)

  proimages = tf.stack([maybe_transform(image) for image in tf.unstack(images)])

  return proimages

def distort_blur(image, blur_selector=0):
  # blur image with selected type of blur
  # image: 3D, float32, in [0,1), numpy array
  # median: 9: good for 2MP, 5: good for 0.5MP
  # bilateral: 75,75: good for 2MP, 35,35: 0.5MP
  # a * (input_res + 1) gives a good approximation of the above for a = 1.4, 25
  assert 0 <= blur_selector <= 1, 'blur_selector outside of bounds.'
  def blur_function(img, blur_selector):
    # img: 3D numpy array
    # blur_selector in [0,1]
    # resolution in MP
    res = np.prod(img.shape[:2])/1_000_000
    random_int = 2*(np.random.randint(0, np.rint(1.4*(res+1)))+1) + 1
    # print('res, random_int', res, random_int)
    if blur_selector == 0:
      # median blur asks the input to have specific type
      img = (img*255).astype(np.uint8)
      return cv2.medianBlur(img, random_int).astype(np.float32)/255
    elif blur_selector == 1:
      sigmaSpace = np.rint(25*(res+1))
      return cv2.bilateralFilter(img, random_int, sigmaSpace, sigmaSpace)
    else:
      assert False, f"Not valid blur selector {blur_selector}."
  blurred = tf.py_func(blur_function, [image, blur_selector], tf.float32)
  blurred.set_shape(image.get_shape())

  return blurred
