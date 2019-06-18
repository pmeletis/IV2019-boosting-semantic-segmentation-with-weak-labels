
import sys
import os.path as op
sys.path.append(op.split(op.split(op.split(op.realpath(__file__))[0])[0])[0])
from datetime import datetime
import copy

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt

from input_pipelines.heterogeneous_supervision.input_functions import train_input

class params(object):
  # height_feature_extractor = 512
  # width_feature_extractor = 512
  # preserve_aspect_ratio = True
  # Nb_vistas = 1
  # Nb_open_images = 2
  # Nb = Nb_vistas
  # tfrecords_path = PATH_MAPIL
  # Ntrain = 18000
  per_pixel_dataset_name = 'cityscapes' # 'vistas' # 
  Nb_per_pixel = 1
  Nb_per_bbox = 2
  training_lids2cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, -1]
  training_problem_def = {'lids2cids': copy.copy(training_lids2cids)}
  training_lids2cids[65] = 65
    
  plotting = True
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
    # print(fe, la)
    proimages = fe['proimages'].numpy()
    prolabels_per_pixel = la['prolabels_per_pixel'].numpy()
    prolabels_per_bbox = la['prolabels_per_bbox'].numpy()
    print(
        proimages.shape,
        prolabels_per_pixel.shape,
        prolabels_per_bbox.shape,
        fe['rawimagespaths'],
        fe['imageids'],
        sep='\n',
        end='\n--------------------------------------\n',
        )
    print(datetime.now() - start)
    start = datetime.now()

    # visual checking
    if params.plotting:
      Nb_total = params.Nb_per_pixel + params.Nb_per_bbox
      proimages = (proimages + 1) / 2
      ims = np.concatenate(np.split(proimages, Nb_total), axis=2)[0]
      prolabels = np.concatenate([prolabels_per_pixel, np.argmax(prolabels_per_bbox, axis=-1)])
      las = np.concatenate(np.split(prolabels, Nb_total), axis=2)[0]
      axs[0].imshow(ims)
      axs[1].imshow(las)
      plt.waitforbuttonpress(timeout=100.0)
