"""Example of how to use Semantic Segmentation system for evaluation.
"""

import sys
import os
import pickle

import numpy as np
np.set_printoptions(formatter={'float': '{:>5.2f}'.format}, nanstr=u'nan', linewidth=10000)

import tensorflow as tf

from system_factory import SemanticSegmentation

from input_pipelines.cityscapes.input_cityscapes import evaluate_input as eval_fn
from input_pipelines.cityscapes.input_cityscapes import add_evaluate_input_pipeline_arguments
# from input_pipelines.vistas.input_vistas import evaluate_input as eval_fn
# from input_pipelines.vistas.input_vistas import add_evaluate_input_pipeline_arguments

from models.resnet50_extended_model_hierarchical import model as model_fn, add_model_arguments
from utils.utils import (SemanticSegmentationArguments, print_metrics_from_confusion_matrix,
                         split_path)


def main(argv):
  ssargs = SemanticSegmentationArguments(mode=tf.estimator.ModeKeys.EVAL)
  add_evaluate_input_pipeline_arguments(ssargs.argparser)
  add_model_arguments(ssargs.argparser)
  ssargs.argparser.add_argument(
      'per_pixel_dataset_name',
      type=str,
      choices=['vistas', 'cityscapes'],
      help='During evaluation, it must be given the training dataset name.')
  # parse given arguments, add extra ones and validate
  args = ssargs.parse_args(argv)
  _add_extra_args(args)

  # vars(args).items() returns (key,value) tuples from args.__dict__
  # and sorted uses first element of tuples to sort
  # args_dict = collections.OrderedDict(sorted(vars(args).items()))
  # print(args_dict)

  settings = args

  system = SemanticSegmentation({'eval': eval_fn}, model_fn, settings)

  labels = system.settings.evaluation_problem_def['cids2labels']
  void_exists = -1 in system.settings.evaluation_problem_def['lids2cids']
  if void_exists and not system.settings.train_void_class:
    labels = labels[:-1]

  all_metrics = system.evaluate()

  ## offline save metrics

  # print full metrics to readable file
  mr_filename = os.path.join(system.settings.eval_res_dir, 'all_metrics.txt')
  with open(mr_filename, 'w') as f:
    for metrics in all_metrics:
      print(f"{metrics['global_step']:>05} ", end='', file=f)
      print_metrics_from_confusion_matrix(metrics['confusion_matrix'], labels, printfile=f)

  # save raw metrics to pickle for future reference
  # TODO: maybe move to system
  m_filename = os.path.join(system.settings.eval_res_dir, 'all_metrics.p')
  with open(m_filename, 'wb') as f:
    pickle.dump(all_metrics, f)

def _add_extra_args(args):
  # disable regularizer and set batch_norm_decay to random value
  # temp solution so as with blocks to work
  args.regularization_weight = 0.0
  args.batch_norm_decay = 1.0
  # args.batch_norm_accumulate_statistics = False # by default its false

  # force disable XLA, since there is an internal TF error till at least r1.4
  # TODO: remove this when error is fixed
  # print('\nXLA is disabled due to internal TF bug. If you want to remove through evaluate.py.\n')
  # args.enable_xla = False

if __name__ == '__main__':
  raise NotImplementedError('Evaluation not yet supported.')
  main(sys.argv[1:])
