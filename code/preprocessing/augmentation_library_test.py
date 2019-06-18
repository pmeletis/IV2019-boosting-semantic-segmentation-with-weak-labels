"""
Tests fro augmentation_library.
(c) Panagiotis Meletis, 2017-2018
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from augmentation_library import (
    random_upscaling, random_downscaling, random_scaling,
    random_color, random_blur, random_flipping)

im_path = '/media/panos/data/datasets/cityscapes/leftImg8bit/val/munster/munster_000000_000019_leftImg8bit.png'
la_path = '/media/panos/data/datasets/cityscapes/gtFine/val/munster/munster_000000_000019_gtFine_color.png'

def random_upscaling_test():
  Nb = 6
  image = np.array(Image.open(im_path))
  images = np.stack([image]*Nb)
  label = np.array(Image.open(la_path))
  label = label[..., 0]
  labels = np.stack([label]*Nb)
  # print(images.dtype, np.max(images), labels.dtype, np.max(labels))
  images = images.astype(np.float32)
  images /= 256
  labels = labels.astype(np.int32)
  # print(images.dtype, np.max(images), labels.dtype, np.max(labels))
  images = tf.convert_to_tensor(images, dtype=tf.float32)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  proimages, prolabels = random_upscaling(images, labels, [1.0, 3.0], enable_assertions=True)

  # code assertions
  # classes
  # assert isinstance(random_factor, tf.Tensor), 'random_factor must be a tf.Tensor.'
  assert isinstance(proimages, tf.Tensor), 'proimages must be a tf.Tensor.'
  assert isinstance(prolabels, tf.Tensor), 'prolabels must be a tf.Tensor.'
  # shapes
  # random_factor.shape.assert_has_rank(0)
  proimages.shape.assert_has_rank(4)
  prolabels.shape.assert_has_rank(3)
  # TODO(panos): make next check dynamic
  assert proimages.shape[1:3] == prolabels.shape[1:3], (
      f"proimages {proimages.shape} and prolabels {prolabels.shape} "
      "must have same spatial size.")
  # types
  assert proimages.dtype == tf.float32, 'proimages must have tf.float32 tf.DType.'
  assert prolabels.dtype == tf.int32, 'prolabels must have tf.int32 tf.DType.'
  # values
  # TODO(panos): add value checks

  # visual checking
  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()
  sess = tf.Session()
  for _ in range(20):
    outputs = sess.run([proimages, prolabels])
    print(outputs[0].shape, outputs[1].shape)
    ims = np.concatenate(np.split(outputs[0], Nb), axis=2)[0]
    las = np.concatenate(np.split(outputs[1], Nb), axis=2)[0]
    print(ims.shape, las.shape)
    axs[0].imshow(ims)
    axs[1].imshow(las)
    plt.waitforbuttonpress(timeout=100.0)

def random_downscaling_test():
  Nb = 6
  image = np.array(Image.open(im_path))
  images = np.stack([image]*Nb)
  label = np.array(Image.open(la_path))
  label = label[..., 0]
  labels = np.stack([label]*Nb)
  # print(images.dtype, np.max(images), labels.dtype, np.max(labels))
  images = images.astype(np.float32)
  images /= 256
  labels = labels.astype(np.int32)
  # print(images.dtype, np.max(images), labels.dtype, np.max(labels))
  images = tf.convert_to_tensor(images, dtype=tf.float32)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  proimages, prolabels = random_downscaling(images, labels, [1.0, 3.0], 255, enable_assertions=True)

  # code assertions
  # classes
  # assert isinstance(random_factor, tf.Tensor), 'random_factor must be a tf.Tensor.'
  assert isinstance(proimages, tf.Tensor), 'proimages must be a tf.Tensor.'
  assert isinstance(prolabels, tf.Tensor), 'prolabels must be a tf.Tensor.'
  # shapes
  # random_factor.shape.assert_has_rank(0)
  proimages.shape.assert_has_rank(4)
  prolabels.shape.assert_has_rank(3)
  # TODO(panos): make next check dynamic
  assert proimages.shape[1:3] == prolabels.shape[1:3], (
      f"proimages {proimages.shape} and prolabels {prolabels.shape} "
      "must have same spatial size.")
  # types
  assert proimages.dtype == tf.float32, 'proimages must have tf.float32 tf.DType.'
  assert prolabels.dtype == tf.int32, 'prolabels must have tf.int32 tf.DType.'
  # values
  # TODO(panos): add value checks

  # visual checking
  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()
  sess = tf.Session()
  for _ in range(20):
    outputs = sess.run([proimages, prolabels])
    print(outputs[0].shape, outputs[1].shape)
    ims = np.concatenate(np.split(outputs[0], Nb), axis=2)[0]
    las = np.concatenate(np.split(outputs[1], Nb), axis=2)[0]
    print(ims.shape, las.shape)
    axs[0].imshow(ims)
    axs[1].imshow(las)
    plt.waitforbuttonpress(timeout=100.0)

def random_scaling_test():
  Nb = 6
  image = np.array(Image.open(im_path))
  images = np.stack([image]*Nb)
  label = np.array(Image.open(la_path))
  label = label[..., 0]
  labels = np.stack([label]*Nb)
  # print(images.dtype, np.max(images), labels.dtype, np.max(labels))
  images = images.astype(np.float32)
  images /= 256
  labels = labels.astype(np.int32)
  # print(images.dtype, np.max(images), labels.dtype, np.max(labels))
  images = tf.convert_to_tensor(images, dtype=tf.float32)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  proimages, prolabels = random_scaling(images, labels, [1.0, 3.0], 255, enable_assertions=True)

  # code assertions
  # classes
  assert isinstance(proimages, tf.Tensor), 'proimages must be a tf.Tensor.'
  assert isinstance(prolabels, tf.Tensor), 'prolabels must be a tf.Tensor.'
  # shapes
  proimages.shape.assert_has_rank(4)
  prolabels.shape.assert_has_rank(3)
  # TODO(panos): make next check dynamic
  assert proimages.shape[1:3] == prolabels.shape[1:3], (
      f"proimages {proimages.shape} and prolabels {prolabels.shape} "
      "must have same spatial size.")
  # types
  assert proimages.dtype == tf.float32, 'proimages must have tf.float32 tf.DType.'
  assert prolabels.dtype == tf.int32, 'prolabels must have tf.int32 tf.DType.'
  # values
  # TODO(panos): add value checks

  # visual checking
  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()
  sess = tf.Session()
  for _ in range(20):
    outputs = sess.run([proimages, prolabels])
    print(outputs[0].shape, outputs[1].shape)
    ims = np.concatenate(np.split(outputs[0], Nb), axis=2)[0]
    las = np.concatenate(np.split(outputs[1], Nb), axis=2)[0]
    print(ims.shape, las.shape)
    axs[0].imshow(ims)
    axs[1].imshow(las)
    plt.waitforbuttonpress(timeout=100.0)

def random_color_test():
  images = np.array(Image.open(im_path))
  images = np.stack([images]*4)
  images = images.astype(np.float32)
  images /= 256
  images = tf.convert_to_tensor(images, dtype=tf.float32)
  print(images.shape)
  proimages = random_color(images)

  # visual checking
  fig = plt.figure(0)
  axs = [None]*4
  for i in range(4):
    axs[i] = fig.add_subplot(2, 2, i+1)
    axs[i].set_axis_off()
  fig.tight_layout()
  sess = tf.Session()
  for _ in range(40):
    outputs = sess.run(proimages)
    print(outputs.shape)
    for i in range(4):
      axs[i].imshow(outputs[i])
    plt.waitforbuttonpress(timeout=100.0)

def random_blur_test():
  images = np.array(Image.open(im_path))
  images = np.stack([images]*4)
  images = images.astype(np.float32)
  images /= 256
  images = tf.convert_to_tensor(images, dtype=tf.float32)
  proimages = random_blur(images)

  # visual checking
  fig = plt.figure(0)
  axs = [None]*4
  for i in range(4):
    axs[i] = fig.add_subplot(2, 2, i+1)
    axs[i].set_axis_off()
  fig.tight_layout()
  sess = tf.Session()
  for _ in range(40):
    outputs = sess.run(proimages)
    for i in range(4):
      axs[i].imshow(outputs[i][:800, :800, :])
    plt.waitforbuttonpress(timeout=100.0)

def random_flipping_test():
  Nb = 6
  image = np.array(Image.open(im_path))
  images = np.stack([image]*Nb)
  label = np.array(Image.open(la_path))
  label = label[..., 0]
  labels = np.stack([label]*Nb)
  images = images.astype(np.float32)
  images /= 256
  labels = labels.astype(np.int32)
  images = tf.convert_to_tensor(images, dtype=tf.float32)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  proimages, prolabels = random_flipping(images, labels)

  # visual checking
  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()
  sess = tf.Session()
  for _ in range(20):
    outputs = sess.run([proimages, prolabels])
    print(outputs[0].shape, outputs[1].shape)
    ims = np.concatenate(np.split(outputs[0], Nb), axis=2)[0]
    las = np.concatenate(np.split(outputs[1], Nb), axis=2)[0]
    print(ims.shape, las.shape)
    axs[0].imshow(ims)
    axs[1].imshow(las)
    plt.waitforbuttonpress(timeout=100.0)

if __name__ == '__main__':
  # random_downscaling_test()
  # random_upscaling_test()
  # random_scaling_test()
  # random_color_test()
  # random_blur_test()
  random_flipping_test()
