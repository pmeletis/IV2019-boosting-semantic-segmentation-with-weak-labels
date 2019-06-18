
import sys
import os.path as op
sys.path.append(op.split(op.split(op.split(op.realpath(__file__))[0])[0])[0])
from datetime import datetime

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt

from input_pipelines.open_images.input_subset_bboxes import train_input

class params(object):
  height_feature_extractor = 512
  width_feature_extractor = 512
  preserve_aspect_ratio = False
  Nb = 8
  plotting = False
  distribute = False

class runconfig(object):
  train_distribute = False

if params.plotting:
  fig = plt.figure(0)
  axs = [None]*2
  axs[0] = fig.add_subplot(2, 1, 1)
  axs[0].set_axis_off()
  axs[1] = fig.add_subplot(2, 1, 2)
  axs[1].set_axis_off()
  fig.tight_layout()

with tf.device('/cpu:0'):
  start = datetime.now()
  for fe, la in train_input(runconfig, params):
    proimages = fe['proimages'].numpy()
    prolabels = la['prolabels'].numpy()
    print(
        # fe['rawimages'].shape,
        # la['rawlabels'].shape,
        proimages.shape,
        prolabels.shape,
        fe['imageids'],
        sep='\n',
        end='\n--------------------------------------\n',
        )
    print(datetime.now() - start)
    start = datetime.now()

    # visual checking
    if params.plotting:
      proimages = (proimages + 1) / 2
      print(proimages.shape, params.Nb)
      ims = np.concatenate(np.split(proimages, params.Nb), axis=2)[0]
      las = np.concatenate(np.split(np.max(prolabels[..., :-1], axis=-1), params.Nb), axis=2)[0]
      axs[0].imshow(ims)
      axs[1].imshow(las)
      plt.waitforbuttonpress(timeout=100.0)
