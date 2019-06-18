from datetime import datetime
import tensorflow as tf
import sys, copy
from os.path import split, realpath
sys.path.append(split(split(split(realpath(__file__))[0])[0])[0])
from input_pipelines.cityscapes.input_cityscapes import train_input
from utils.utils import _replacevoids
import matplotlib.pyplot as plt
import numpy as np

_PATH_CITYS = ('/media/panos/data/datasets/cityscapes'
               '/tfrecords/trainFine_v5.tfrecords')

def train_input_test():
  # !!! IMPORTANT !!!
  # cannot use e.g. features['rawimages'].eval() and features['rawlabels'].eval()
  # because every eval causes the required nodes to be computed again and each eval
  # reads new examples...
  sess = tf.InteractiveSession()
  class params(object):
    height_feature_extractor = 1024
    width_feature_extractor = 2048
    preserve_aspect_ratio = False
    # Ntrain = 2975
    tfrecords_path = _PATH_CITYS
    Nb = 12
    training_lids2cids = [-1, 0, -1, -1, -1, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, -1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    training_problem_def = {'lids2cids': copy.copy(training_lids2cids)}
    training_lids2cids = _replacevoids(training_lids2cids)
    plotting = False
    distribute = False

  class runconfig(object):
    train_distribute = False

  with tf.device('/cpu:0'):
    dataset = train_input(runconfig, params)
    features, labels = dataset.make_one_shot_iterator().get_next()

  print(features, labels)

  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()

  for i in range(1000):
    start = datetime.now()
    fe, la = sess.run((features, labels))
    print(
        fe['rawimages'].shape,
        la['rawlabels'].shape,
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
      axs[0].imshow(ims)
      axs[1].imshow(las)
      plt.waitforbuttonpress(timeout=100.0)

def main():
  train_input_test()
  return

if __name__ == '__main__':
  main()