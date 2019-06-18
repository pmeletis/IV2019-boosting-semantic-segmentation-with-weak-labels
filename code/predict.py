"""
Example of how to use Semantic Segmentation system for prediction.
"""

import sys
import os
import os.path as op
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
plt.ion() # for expected waitforbuttonpress functionality
from PIL import Image
import tensorflow as tf

from system_factory import SemanticSegmentation
from input_pipelines.dataset_agnostic.dataset_agnostic_predict_input import (
    predict_input as predict_fn, add_predict_input_pipeline_arguments)
from models.resnet50_extended_model_hierarchical import model as model_fn, add_model_arguments
from utils.utils import SemanticSegmentationArguments, split_path

def main(argv):
  tf.logging.set_verbosity(tf.logging.WARN)
  # add all related arguments
  ssargs = SemanticSegmentationArguments(mode=tf.estimator.ModeKeys.PREDICT)
  add_predict_input_pipeline_arguments(ssargs.argparser)
  add_model_arguments(ssargs.argparser)
  ssargs.argparser.add_argument(
      'per_pixel_dataset_name',
      type=str,
      choices=['vistas', 'cityscapes'],
      help='During evaluation, it must be given the training dataset name.')
  _add_predict_arguments(ssargs.argparser)
  # parse given arguments, add extra ones and validate
  settings = ssargs.parse_args(argv)
  _add_extra_args(settings)
  _validate_settings(settings)

  # TODO(panos): move to system
  if settings.results_dir and not op.exists(settings.results_dir):
    os.makedirs(settings.results_dir)

  system = SemanticSegmentation({'predict': predict_fn}, model_fn, settings)
  settings = system.settings

  # TODO: 2D plotting not needed since estimator outputs one example
  # per step irrespective of settings.Nb
  # TODO(panos): change Nb name so as not to be confusing
  Nb = 1 # not the same as settings.Nb
  Np = 2 # number of plots for each batch example
  if settings.plot_l1_confidence or settings.plot_l2_confidence:
    Np += 1

  if settings.plotting:
    fig = plt.figure(0)
    # figure colorbars
    class _CB(object):
      # dummy class so we don't need to check existance of cb when we use it
      def remove(self):
        pass
    cb = _CB()
    axs = [[None for j in range(Np)] for i in range(Nb)]
    for i in range(Nb):
      for j in range(Np):
        axs[i][j] = fig.add_subplot(Np, Nb, Np*i+(j+1))
        axs[i][j].set_axis_off()
    fig.tight_layout()
  elif settings.plotting_overlapped:
    fig = plt.figure(0)
    axs = [None for i in range(Nb)]
    for i in range(Nb):
      axs[i] = fig.add_subplot(1, Nb, i+1)
      axs[i].set_axis_off()
    fig.tight_layout()

  # color mappings for outputs
  if settings.plotting or settings.plotting_overlapped:
    palettes = np.array(settings.inference_problem_def['cids2colors'], dtype=np.uint8)
  if settings.export_lids_images:
    idspalettes = np.array(settings.inference_problem_def['cids2lids'], dtype=np.uint8)
  if settings.export_color_decisions or settings.export_overlapped_color_decisions:
    colorpalettes = np.array(settings.inference_problem_def['cids2colors'], dtype=np.uint8)

  if settings.Nb > 1:
    tf.logging.warn('Timings are valid only for Nb=1.')
  start = datetime.now()
  total_accumulator = datetime.now()
  for outputs in system.predict():
    # keep next two lines first in the loop for correct timings and don't print
    # anything else inside the loop for single line refresing
    total_accumulator += datetime.now()-start
    sys.stdout.write(f'Time per image (input pipeline + network): {datetime.now()-start}\r')
    sys.stdout.flush()

    # unpack outputs
    # decs: 2D, np.int32, in [0, Nclasses-1]
    # rawimage: 3D, np.uint8
    # rawimagepath: 0D, bytes object, convert to str object (using str()) for using it
    # probs: 3D, np.float32, in [0, 1]
    decs = outputs['decisions']
    rawimage = outputs['rawimages']
    rawimagepath = str(outputs['rawimagespaths'])
    l1_probs = outputs['l1_probabilities']
    l2_probs = outputs['l2_vehicle_probabilities']

    # live plotting
    if settings.plotting:
      for i in range(Nb):
        axs[i][0].imshow(rawimage)
        axs[i][1].imshow(palettes[decs])
        if settings.plot_l2_confidence or settings.plot_l1_confidence:
          # nipy_spectral colormap has high contrast at high values
          # for high contrast at high values we plot prob^50
          # probs^{10, 50, 100}: cut-off values: {0.6, 0.9, 0.95}
          conf = axs[i][2].imshow(
              np.concatenate(
                  list(map(lambda p: np.amax(np.power(p, 50), axis=2), [l1_probs, l2_probs])),
                  axis=1),
              cmap='nipy_spectral')
          # axs[i][2].imshow(np.amax(probs, axis=2), cmap='Greys_r')
          # remove colorbar from figure, cause in loop a new colorbar is added each step
          cb.remove()
          cb = fig.colorbar(conf, ax=axs[i][2], ticks=[])

      plt.waitforbuttonpress(timeout=settings.timeout)

    # live plotting overlapped
    if settings.plotting_overlapped:
      result_col = palettes[decs]
      alpha = 0.5
      overlapped = ((alpha*rawimage + (1-alpha)*result_col)).astype(np.uint8)
      for i in range(Nb):
        axs[i].imshow(overlapped)
      fig.tight_layout()
      plt.waitforbuttonpress(timeout=settings.timeout)

    # export label ids images
    if settings.export_lids_images:
      result_ids = idspalettes[decs]
      out_fname = os.path.join(
          settings.results_dir, split_path(rawimagepath)[1] + '_result_lids.png')
      # if you want to overwrite files comment next line
      assert not os.path.exists(out_fname), 'Output filename ({out_fname}) already exists.'
      Image.fromarray(result_ids).save(out_fname)

    # export color images
    if settings.export_color_decisions:
      result_col = colorpalettes[decs]
      out_fname = os.path.join(
          settings.results_dir, split_path(rawimagepath)[1] + '_result_color.png')
      # if you want to overwrite files comment next line
      assert not os.path.exists(out_fname), 'Output filename ({out_fname}) already exists.'
      Image.fromarray(result_col).save(out_fname)

    # export overlapped color ids images and input images
    if settings.export_overlapped_color_decisions:
      result_col = colorpalettes[decs]
      alpha = 0.5
      overlapped = ((alpha*rawimage + (1-alpha)*result_col)).astype(np.uint8)
      out_fname = op.join(
          settings.results_dir, split_path(rawimagepath)[1] + '_result_overlapped_color.png')
      # if you want to overwrite files comment next line
      assert not op.exists(out_fname), 'Output filename ({out_fname}) already exists.'
      Image.fromarray(overlapped).save(out_fname)

    # keep next line last for correct timings
    start = datetime.now()

  print('\nTotal time (input pipeline + network):', datetime.now() - total_accumulator)

def _add_predict_arguments(argparser):
  # TODO(panos): make next two options mutually exclusive
  argparser.add_argument('--plotting', action='store_true',
                         help='Whether to plot results.')
  argparser.add_argument('--plotting_overlapped', action='store_true',
                         help='Whether to plot overlapped segmentation results on input images.')
  argparser.add_argument('--plot_l1_confidence', action='store_true',
                         help='Whether to plot confidence of predictions, i.e. the MAP of the '
                              'per-pixel output class. Please note that the value of confidences '
                              'are streched and don\'t correspond to original probabilities of '
                              'the predicted classes.')
  argparser.add_argument('--plot_l2_confidence', action='store_true',
                         help='Whether to plot confidence of predictions, i.e. the MAP of the '
                              'per-pixel output class. Please note that the value of confidences '
                              'are streched and don\'t correspond to original probabilities of '
                              'the predicted classes.')
  argparser.add_argument('--timeout', type=float, default=10.0,
                         help='Timeout for continuous plotting, if plotting flag is provided.')
  argparser.add_argument('--export_color_decisions', action='store_true',
                         help='Whether to export color image results.')
  argparser.add_argument('--export_overlapped_color_decisions', action='store_true',
                         help='Whether to export overlapped color images and decisions results.')
  argparser.add_argument('--export_lids_images', action='store_true',
                         help=('Whether to export label ids image results. Label ids are defined '
                               'in {training,inference}_problem_def_path.'))
  argparser.add_argument('--results_dir', type=str, default=None,
                         help='If provided results will be written to this directory.')

def _add_extra_args(settings):
  # SemanticSegmentation system expects also the following arguments to be provided by the user
  # disable regularizer and set batch_norm_decay to random value for argument contexts to work
  settings.regularization_weight = 0.0
  settings.batch_norm_decay = 1.0
  # settings.batch_norm_accumulate_statistics = False # by default its false

  # request which output should be returned by SemanticSegmentation.predict()
  # prediction keys are defined in models and input_pipelines
  # e.g. 'decisions', 'probabilities' are defined in the
  #   resnet50_extended_model `predictions` output
  # and 'rawimages', 'rawimagespaths' are defined in the cityscapes train input pipeline
  settings.predict_keys = ['decisions', 'l1_probabilities', 'l2_vehicle_probabilities', 'rawimages', 'rawimagespaths']

def _validate_settings(settings):
  if hasattr(settings, 'export_lids_images') and hasattr(settings, 'export_color_decisions'):
    if settings.export_lids_images or settings.export_color_decisions:
      assert settings.results_dir is not None and op.isdir(settings.results_dir), (
          'results_dir must a valid path if export_{lids, color}_images flags are True.')

if __name__ == '__main__':
  main(sys.argv[1:])
