"""
Inference input pipeline that is dataset agnostic.
This essentially means that a model can predict in arbitrary image dimensions.
"""
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
import sys
import os
from os.path import join, split, realpath
sys.path.append(split(split(split(realpath(__file__))[0])[0])[0])
from input_pipelines.utils import from_0_1_to_m1_1
from tensorflow.python.util.deprecation import deprecated
from utils.utils import resize_images_or_labels

@deprecated('2018-07-09',
            'Please use resize_images_or_labels from utils/utils.py.')
def _resize_respecting_aspect_ratio(features, target_size, method):
  """
  Resize `features` spatial dimensions to `target_size` respecting the aspect ratio.
  Resizing aims at producing the closer spatial size to 'target_size' respecting the aspect ratio.
  Thus, the output features most of the times have different spatial size than `target_size`.
  If the aspect ratios of features and target_size are the same then features are resized to target size.
  Otherwise, at least one of dimensions will have the same length as in the target_size.

  This function is intended for resizing images and labels for semantic segmentation.

  The algorithm computes the aspect ratio of `features` spatial diamensions (H/W) and
    of `target_size` (h/w) and:
      if h/w < H/W: rescale with (w/W) * [H, W] -> [(w/W)*H, w], which assures that (w/W)*H > h
      if h/w > H/W: rescale with (h/H) * [H, W] -> [h, (h/H)*W], which assures that (h/H)*W > w
    this results in the smaller spatial dimensions that are bigger than initial spatial dimensions.

  Arguments:
    features: tf.float32 and (?, ?, ?, ?), or tf.uint32 and (?, ?, ?)
    target_size: Python tuple

  Comments:
    spatial size of `features` is of format Nb x H x W [x C]

  Output:
    rescaled features using tf.image.resize_images
  """

  is_image = features.dtype == tf.float32 and features.shape.ndims == 4
  is_label = features.dtype == tf.uint32 and features.shape.ndims == 3

  assert is_image or is_label, "'features' don't have the correct specification."

  features_size = tf.shape(features)[1:3]
  H, W = features_size[0], features_size[1]
  h, w = target_size

  target_size_ratio = h/w
  features_size_ratio = tf.cast(H/W, tf.float32)
  pred_fn_pairs = {
      tf.less(target_size_ratio, features_size_ratio): lambda: tf.cast(w/W, tf.float32),
      tf.greater(target_size_ratio, features_size_ratio): lambda: tf.cast(h/H, tf.float32)}
  factor = tf.case(pred_fn_pairs,
                   default=lambda: tf.constant(1.0, dtype=tf.float32))

  # new_size ceiling and the crop, because of float32 to int32 rounding errors
  new_size = tf.cast(tf.ceil(factor*tf.cast(features_size, tf.float32)), tf.int32)
  # new_size = tf.Print(new_size, [new_size], message='debug: ')
  if features.shape.ndims == 4:
    resized_features = tf.image.resize_images(features, new_size, method)
  elif features.shape.ndims == 3:
    resized_features = tf.image.resize_images(features[..., tf.newaxis], new_size, method)[..., 0]
  else:
    print('Wrong input rank.')
  # crop
  pred_fn_pairs = {tf.less(target_size_ratio, features_size_ratio): lambda: resized_features[:, :, :w, ...],
                   tf.greater(target_size_ratio, features_size_ratio): lambda: resized_features[:, :h, :, ...]}
  resized_features = tf.case(pred_fn_pairs,
                             default=lambda: features)

  # some sanity checks
  resized_features_shape = resized_features.shape
  if resized_features_shape[1:2].is_fully_defined():
    assert resized_features_shape[1] == h
  else:
    pass
    # print('\nassertion for height not done\n')
  if resized_features_shape[2:3].is_fully_defined():
    assert resized_features_shape[2] == w
  else:
    pass
    # print('\nassertion for width not done\n')

  return resized_features

def _predict_image_generator(params):
  SUPPORTED_EXTENSIONS = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'ppm', 'PPM']
  fnames = []
  for se in SUPPORTED_EXTENSIONS:
    glob_path = join(params.predict_dir, '**', '*.' + se)
    fnames.extend(glob.glob(glob_path, recursive=True))
  print(f"Found {len(fnames)} images.")

  def _check_specs(im):
    # make sure we only read RGB images as an extra fail-safe
    return im.mode == 'RGB'

  for im_fname in fnames:
    # start = datetime.now()
    im = Image.open(im_fname)
    # next line is time consuming (can take up to 400ms for im of 2 MPixels)
    # and apparently is not needed
    # im_array = np.array(im)
    # print('reading time:', datetime.now() - start)
    if not _check_specs(im):
      print(f"{im_fname} [{im}] didn\'t comply with specs. Trying to transform it, otherwise it will ignore it.")
      if im.mode in ['L', 'P', 'RGBA']:
        im = im.convert(mode='RGB')
    yield im, im_fname.encode('utf-8'), im.height, im.width

def _predict_preprocess(image, params):
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = resize_images_or_labels(image[tf.newaxis, ...],
                                  (params.height_feature_extractor, params.width_feature_extractor),
                                  tf.image.ResizeMethod.BILINEAR,
                                  preserve_aspect_ratio=params.preserve_aspect_ratio,
                                  mode='max')[0]
  # model expects input in [-1, 1)
  proimage = from_0_1_to_m1_1(image)

  return proimage

def predict_input(config, params):
  del config
  dataset = tf.data.Dataset.from_generator(lambda: _predict_image_generator(params),
                                           output_types=(tf.uint8, tf.string, tf.int32, tf.int32),
                                           output_shapes=((None, None, None), (), (), ()))
  # drop height and width since are not used
  dataset = dataset.map(lambda im, im_path, height, width: (
      im_path, im, _predict_preprocess(im, params)), num_parallel_calls=15)
  # TODO(panos): bring this back when batching with different input sizes is possible
  # dataset = dataset.batch(params.Nb)
  if params.Nb > 1:
    print('\n\nBatching for inference is disabled (in case input images don\'t have the same size).\n\n')
  dataset = dataset.batch(1)

  def _grouping(imp, rim, pim):
    # group dataset elements as required by estimator
    features = {'rawimages': rim,
                'proimages': pim,
                'rawimagespaths': imp}
    return features

  dataset = dataset.map(_grouping, num_parallel_calls=10)
  dataset = dataset.prefetch(None)

  return dataset

def add_predict_input_pipeline_arguments(argparser):
  """
  Add arguments required by the input pipeline.

  Arguments:
    argparser: an argparse.ArgumentParser object to add arguments
  """
  argparser.add_argument('--preserve_aspect_ratio', action='store_true',
                         help='Smartly resizes the input image respecting the aspect ratio.')
