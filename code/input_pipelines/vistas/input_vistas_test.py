from datetime import datetime
import tensorflow as tf
import sys, copy
from os.path import split, realpath
sys.path.append(split(split(split(realpath(__file__))[0])[0])[0])
import matplotlib.pyplot as plt
import numpy as np

from input_pipelines.vistas.input_vistas import _parse_tfexample, train_input

PATH_MAPIL = ('/media/panos/data/datasets/mapillary'
              '/mapillary-vistas-dataset_public_v1.0/tfrecords/train.tfrecord')

def parse_tfexample_test():
  tf.InteractiveSession()
  cnt = 0
  for str_rec in tf.python_io.tf_record_iterator(PATH_MAPIL):
    image, label, im_path, la_path = map(tf.Tensor.eval, _parse_tfexample(str_rec))
    print(image.shape, label.shape, im_path, la_path)
    cnt += 1
    if cnt == 10:
      break

def train_input_test():
  sess = tf.InteractiveSession()
  class params(object):
    height_feature_extractor = 827
    width_feature_extractor = 1139
    preserve_aspect_ratio = False
    tfrecords_path = PATH_MAPIL
    Nb = 16
    Ntrain = 18000
    training_lids2cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, -1]
    training_problem_def = {'lids2cids': copy.copy(training_lids2cids)}
    training_lids2cids[65] = 65
    distribute = False
    plotting = False

  class runconfig(object):
    train_distribute = False

  with tf.device('/cpu:0'):
    dataset = train_input(runconfig, params)
    features, labels = dataset.make_one_shot_iterator().get_next()

  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()

  print(features, labels)
  for i in range(10000):
    start = datetime.now()
    fe, la = sess.run((features, labels))
    print(
        # fe['rawimages'].shape,
        # la['rawlabels'].shape,
        fe['proimages'].shape,
        la['prolabels'].shape,
        fe['rawimagespaths'],
        fe['rawlabelspaths'],
        sep='\n',
        end='\n--------------------------------------\n',
        )
    print(i, datetime.now()-start)

    # visual checking
    if params.plotting:
      fe['proimages'] = (fe['proimages'] + 1) / 2
      ims = np.concatenate(np.split(fe['proimages'], params.Nb), axis=2)[0]
      las = np.concatenate(np.split(la['prolabels'], params.Nb), axis=2)[0]
      # print(ims.shape, las.shape)
      axs[0].imshow(ims)
      axs[1].imshow(las)
      plt.waitforbuttonpress(timeout=100.0)

def main():
  train_input_test()
  return

if __name__ == '__main__':
  main()
