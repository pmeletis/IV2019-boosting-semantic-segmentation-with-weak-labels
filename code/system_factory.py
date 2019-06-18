"""
Semantic Segmentation system.
"""

import copy
import functools
import glob
import collections
import json
from datetime import datetime
from operator import itemgetter
from os.path import join, isdir, split, exists, isdir
from os import makedirs
import numpy as np
from PIL import Image

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

from estimator.define_estimator_hierarchical import define_estimator, _replace_voids, _map_predictions_to_new_cids
from estimator.define_savers import predict_saver, export_frozen_graph_saver
from utils.utils import _replacevoids, print_metrics_from_confusion_matrix

__version__ = '0.9'

class SemanticSegmentation(object):
  """A Semantic Segmentation system class.

  This class uses the tf.estimator API for reproducibility and transferability
  of experiments, as well as to take advantage of automatic parallelization
  (across CPUs, GPUs) that this API provides.

  Components of this class:
    * model for per-pixel semantic segmentation: provided by model_fn
    * input functions for different modes: provided by input_fns
    * settings for the whole system: provided by settings

  Provides the following functions to the user:
    * train
    * evaluate
    * predict

  More info to be added soon...

  """

  # TODO: during prediction and evaluation void decisions (if void exist), i.e. pixels labeled
  # with the last class, should be assigned with the second most probable label, and the same
  # also for metrics and summaries during training.

  def __init__(self, input_fns, model_fn, settings):
    """Constructs a SS instance... (More info to be added...)

    Args:
      input_fns: a dictionary containing 'train', 'eval' and 'predict' keys to corresponding
        input functions. The input functions will be called by the respective functions of this
        class with the following signature: (customconfig, params) and should
        return a tuple of (features, labels) containing feature and label dictionaries with
        the needed str-tf.Tensor pairs. (required keys will be added, for now check code...).
      model_fn: model function for the fully convolutional semantic segmentation model chosen.
        It will be called with the following signature: (mode, features, labels, config, params).
      settings: an object containing all parsed parameters from command line as attributes.

    Comments / design choices:
      1) Everytime a method {train, predict, evaluate} of this object is called a new local
         estimator is created with the desired properties saved in this class' members, specified
         to the respective action. This choice is made purely for memory efficiency.
    """

    # TODO: add input_fns, model_fn and settings checks
    assert settings is not None, ('settings must be provided for now.')

    self._settings = copy.deepcopy(settings)

    self._settings.height_network = self._settings.height_feature_extractor
    self._settings.width_network = self._settings.width_feature_extractor

    with open(self._settings.training_problem_def_path, 'r') as fp:
      self._settings.training_problem_def = json.load(fp)

    # during inference only
    if hasattr(self._settings, 'inference_problem_def_path'):
      if self._settings.inference_problem_def_path:
        with open(self._settings.inference_problem_def_path, 'r') as fp:
          self._settings.inference_problem_def = json.load(fp)
      else:
        # TODO: implement the use of inference problem_def for training results in all functions
        self._settings.inference_problem_def = self._settings.training_problem_def

    # during evaluation only
    if hasattr(self._settings, 'evaluation_problem_def_path'):
      if self._settings.evaluation_problem_def_path:
        with open(self._settings.evaluation_problem_def_path, 'r') as fp:
          self._settings.evaluation_problem_def = json.load(fp)
      else:
        self._settings.evaluation_problem_def = self._settings.training_problem_def


    _set_defaults(self._settings)  
    _validate_settings(self._settings)

    self._input_fns = input_fns
    self._model_fn = model_fn
    self._estimator = None
    # self._warm_start_settings = None
    # _predictor_* are used only for predict_image
    self._predictor_session = None
    self._predictor_session_callable = None

    # configure warm start settings
    # if hasattr(self._settings, 'var_name_to_prev_var_name'):
    #   # start from scratch or continue from log_dir
    #   ckpt_to_initialize_from = self._settings.log_dir
    #   # an empty string '' is False
    #   if self._settings.init_ckpt_path:
    #     ## initialize from checkpoint, e.g. trained on ImageNet
    #     ckpt_to_initialize_from = self._settings.init_ckpt_path
    #   self._warm_start_settings = tf.estimator.WarmStartSettings(
    #       ckpt_to_initialize_from, vars_to_warm_start=None, var_name_to_prev_var_name)

    # _validate_problem_config(self.args.config)

    lids2cids_training = self._settings.training_problem_def['lids2cids']
    self._settings.lids_training_contain_unlabeled = -1 in lids2cids_training
    # +1 since by convention class ids start at 0
    # ATTENTION!: the next formula holds only for not replaced -1 in lids2cids
    self._settings.output_Nclasses = (
        max(lids2cids_training) + 1 +
        (self._settings.lids_training_contain_unlabeled or self._settings.train_void_class))

    # A SemanticSegmentation system is characterized by the classes it's trained to infer.
    #   Internally the system in every mode (train, predict, evaluate) has a mapping for
    #   training to inference
    # TODO(panos): remove *_lids2cids attributes from settings, since don't
    #   characterize system directly, also training_lids2cids are only relevant during training
    # TODO(panos): remove the replacevoids from lids2cids
    if hasattr(self._settings, 'inference_problem_def'):
      if 'training_cids2inference_cids' in self._settings.inference_problem_def.keys():
        self._settings.training_cids2inference_cids = \
            self._settings.inference_problem_def['training_cids2inference_cids']
      else:
        tcids2pcids = list(range(self._settings.output_Nclasses))
        # ignore void class if not trained with losses from void class
        if self._settings.lids_training_contain_unlabeled and not self._settings.train_void_class:
          tcids2pcids[-1] = -1
        self._settings.training_cids2inference_cids = tcids2pcids
    if hasattr(self._settings, 'evaluation_problem_def'):
      if 'training_cids2evaluation_cids' in self._settings.evaluation_problem_def.keys():
        self._settings.training_cids2evaluation_cids = \
            self._settings.evaluation_problem_def['training_cids2evaluation_cids']
      else:
        tcids2ecids = list(range(self._settings.output_Nclasses))
        # ignore void class if not trained with losses from void class
        if self._settings.lids_training_contain_unlabeled and not self._settings.train_void_class:
          tcids2ecids[-1] = -1
        self._settings.training_cids2evaluation_cids = tcids2ecids

    # construct candidate path for evaluation results directory in log directory,
    # with a unique counter index, e.g. if in log_dir/eval there exist
    # eval00, eval01, eval02, eval04 dirs it will create a new dir named eval05
    # TODO: better handle and warn for assumptions
    # for now it assums that only eval_ with 2 digits are present
    existing_eval_dirs = list(filter(isdir, glob.glob(join(self._settings.log_dir, 'eval_*'))))
    if existing_eval_dirs:
      existing_eval_dirs_names = [split(ed)[1] for ed in existing_eval_dirs]
      max_cnt = max([int(edn[-2:]) for edn in existing_eval_dirs_names])
    else:
      max_cnt = -1
    eval_res_dir = join(self._settings.log_dir, 'eval_' + f"{max_cnt + 1:02}")
    # save to settings for external access
    self._settings.eval_res_dir = eval_res_dir

  @property
  def settings(self):
    return self._settings

  def _create_estimator(self, runconfig, warm_start_from=None):
    self._estimator = tf.estimator.Estimator(
        functools.partial(define_estimator, model_fn=self._model_fn),
        model_dir=self._settings.log_dir,
        config=runconfig,
        params=self._settings,
        # warm_start_from=self._warm_start_settings,
        )

    return self._estimator

  def train(self):
    """Train the Semantic Segmentation model.
    """

    # for piecewise_constant learning_rate_schedule
    # TODO(panos): implement patch-wise training in case of patch-training of
    #   feature extractor *_network != *_feature_extractor and the
    #   effective number of examples increase
    self._settings.num_examples_per_epoch = int(
        self._settings.Ntrain *
        self._settings.height_network//self._settings.height_feature_extractor *
        self._settings.width_network//self._settings.width_feature_extractor) # per epoch
    # TODO(panos): as noted in MirrorStrategy, in a multi-gpu setting the effective
    #   batch size is num_gpus * Nb, so deal with this until per batch broadcasting
    #   is implemented in core tensorflow package
    # by default all available gpus of the machine are used
    # TODO(panos): num_gpus should divide Nb
    # Nb_temp = args.Nb // context.num_gpus()
    self._settings.num_batches_per_epoch = int(self._settings.num_examples_per_epoch / self._settings.Nb)
    self._settings.num_training_steps = int(self._settings.Ne * self._settings.num_batches_per_epoch) # per training

    # for piecewise_constant learning_rate_schedule
    # optimizer needs learning rate boundaries and learning rate values
    #learning_rate_decay learning_rate_values
    if self._settings.learning_rate_schedule == 'piecewise_constant':
      # set default if none provided
      if not (self._settings.learning_rate_decay or self._settings.learning_rate_values):
        self._settings.learning_rate_decay = 0.5
      # fix the length of boundaries: should be one less than the values according to tf.piecewise_constant
      last_boundary = self._settings.Ne - self._settings.learning_rate_boundaries[-1]
      if last_boundary == 0:
        self._settings.learning_rate_boundaries.pop()
      elif last_boundary <0:
        raise ValueError('Ne is less than learning rate boundaries.')
      # create an attribute so learning rate boundaries are also logged in epochs
      self._settings.learning_rate_boundaries_epochs = self._settings.learning_rate_boundaries
      # convert from epochs to batched steps
      self._settings.learning_rate_boundaries = [
          lrb * self._settings.num_batches_per_epoch for lrb in self._settings.learning_rate_boundaries]
      # create learning rate values if not explicitly provided
      if self._settings.learning_rate_decay:
        decay_steps = len(self._settings.learning_rate_boundaries) + 1
        self._settings.learning_rate_values = [
            self._settings.learning_rate_initial * self._settings.learning_rate_decay**i
            for i in range(decay_steps)]

    # disable emas for distributed training
    if self._settings.distribute:
      print('\n\nDisabling moving running averages for distributed training.\n\n')
      self._settings.ema_decay = 0

    # create log dir
    if not tf.gfile.Exists(self._settings.log_dir):
      tf.gfile.MakeDirs(self._settings.log_dir)
      print('Created new logging directory:', self._settings.log_dir)

    # set save_checkpoints_steps if not provided
    if not self._settings.save_checkpoints_steps:
      # save one checkpoint per epoch
      self._settings.save_checkpoints_steps = self._settings.num_batches_per_epoch

    # vars(args).items() returns (key,value) tuples from args.__dict__
    # and sorted uses first element of tuples to sort
    settings_dict = collections.OrderedDict(sorted(vars(self._settings).items()))

    # write configuration for future reference
    settings_filename = join(self._settings.log_dir, 'settings.txt')
    assert not exists(settings_filename), (
        f"Previous settings.txt found in "
        f"{self._settings.log_dir}. Rename or delete it manually and restart training.")
    with open(settings_filename, 'w') as f:
      for k, v in enumerate(settings_dict):
        print(f"{k:2} : {v} : {settings_dict[v]}", file=f)

    # define the session_config
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    # session_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
    # session_config.log_device_placement = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session_config = tf.ConfigProto()
    # allow memory growth so not all memory is allocated from the beginning
    session_config.gpu_options.allow_growth = True
    # XLA
    if self._settings.enable_xla:
      session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # some ops (e.g.) assert are not possible to be placed on gpu
    # TODO(panos): remove this when TF error is fixed
    if self._settings.distribute:
      session_config.allow_soft_placement = True
    # define the distribution stategy
    if self._settings.distribute:
      distribution_strategy = tf.contrib.distribute.MirroredStrategy()
    else:
      distribution_strategy = None
      # distribution_strategy = tf.contrib.distribute.OneDeviceStrategy('/GPU:0')

    # Tensorflow internal error till at least r1.4
    # if keep_checkpoint_max is set to 0 or None doesn't do what it is supposed to do from docs
    runconfig = tf.estimator.RunConfig(
        model_dir=self._settings.log_dir,
        save_summary_steps=self._settings.save_summaries_steps,
        save_checkpoints_steps=self._settings.save_checkpoints_steps,
        session_config=session_config,
        keep_checkpoint_max=1_000_000, # some big number to keeps all checkpoints
        log_step_count_steps=self._settings.save_summaries_steps,
        train_distribute=distribution_strategy,
        )

    # create a local estimator
    self._create_estimator(runconfig)

    return self._estimator.train(
        input_fn=self._input_fns['train'],
        max_steps=self._settings.num_training_steps)

  def predict(self):
    if self._settings.Nb > 1:
      print('\nWARNING: during prediction only images with same shape (size and channels) '
            'are supported for batch size greater than one. In case of runtime error '
            'change batch size to 1.\n')
    session_config = tf.ConfigProto()
    # allow memory growth so not all memory is allocated from the beginning
    session_config.gpu_options.allow_growth = True
    if self._settings.enable_xla:
      session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    runconfig = tf.estimator.RunConfig(
        model_dir=self._settings.log_dir,
        session_config=session_config)

    self._create_estimator(runconfig)

    predict_keys = copy.deepcopy(self._settings.predict_keys)

    # maybe TF internal error, predict_keys should be deep copied internally
    predictions = self._estimator.predict(
        input_fn=self._input_fns['predict'],
        predict_keys=predict_keys,
        # if None latest checkpoint in self._settings.model_dir will be used
        checkpoint_path=self._settings.ckpt_path)

    # resize to system dimensions for the outputs
    # predictions = self._resize_predictions(predictions)

    return predictions

  def evaluate(self):

    # number of examples per epoch
    self._settings.num_examples = int(self._settings.Neval *
                            self._settings.height_network//self._settings.height_feature_extractor *
                            self._settings.width_network//self._settings.width_feature_extractor)
    self._settings.num_batches_per_epoch = int(self._settings.num_examples / self._settings.Nb)
    self._settings.num_eval_steps = int(self._settings.num_batches_per_epoch * 1) # 1 epoch

    eval_res_dir = self._settings.eval_res_dir
    print(f"\nWriting results in {eval_res_dir}.\n")
    makedirs(eval_res_dir)

    # write configuration for future reference
    # TODO: if settings.txt exists and train.py is re-run with different settings
    #   (e.g. logging settings) it is overwritten... (probably throw error)
    # if not tf.gfile.Exists(evalres_dir):
    #     tf.gfile.MakeDirs(evalres_dir)
    if exists(join(eval_res_dir, 'settings.txt')):
      print(f"WARNING: previous settings.txt in {eval_res_dir} is ovewritten.")
    with open(join(eval_res_dir, 'settings.txt'), 'w') as f:
      for k, v in vars(self._settings).items():
        print(f"{k} : {v}", file=f)

    session_config = tf.ConfigProto()
    # allow memory growth so not all memory is allocated from the beginning
    session_config.gpu_options.allow_growth = True
    if self._settings.enable_xla:
      session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    runconfig = tf.estimator.RunConfig(
        model_dir=self._settings.log_dir,
        session_config=session_config)

    # create a local estimator
    self._create_estimator(runconfig)

    # get labels needed for online printing
    labels = self._settings.evaluation_problem_def['cids2labels']
    void_exists = -1 in self._settings.evaluation_problem_def['lids2cids']
    if void_exists and not self._settings.train_void_class:
      labels = labels[:-1]

    # evaluate given checkpoint or all checkpoints in log_dir
    # Note: eval_all_ckpts flag has priority over given checkpoint
    all_model_checkpoint_paths = [self._settings.ckpt_path]
    if self._settings.eval_all_ckpts:
      # sanity assignment
      self._settings.ckpt_path = None
      checkpoint_state = tf.train.get_checkpoint_state(self._settings.log_dir)
      # convert from protobuf repeatedScalarFieldContainer to list
      all_model_checkpoint_paths = list(checkpoint_state.all_model_checkpoint_paths)
      print(f"\n{len(all_model_checkpoint_paths)} checkpoint(s) will be evaluated.\n")

    all_metrics = []
    for cp in all_model_checkpoint_paths:
      # metrics contains only confusion matrix for now (and loss and global step)
      metrics = self._estimator.evaluate(
          input_fn=self._input_fns['eval'],
          steps=self._settings.num_eval_steps,
          # if None latest in model_dir will be used
          checkpoint_path=cp,
          name=split(eval_res_dir)[1][-2:])

      # deal with void in evaluation lids2cids
      if (-1 in self._settings.evaluation_problem_def['lids2cids']
          and not self._settings.train_void_class):
        assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
            'internal error: only confusion matrix metric is supported for mapping to '
            'a new problem definition for now. Change to training problem definition.')
        metrics['confusion_matrix'] = metrics['confusion_matrix'][:-1, :-1]

      # online print the summary of metrics to terminal
      print_metrics_from_confusion_matrix(metrics['confusion_matrix'], labels, printcmd=True)

      all_metrics.append(metrics)

    return all_metrics

def _set_defaults(settings):
  # for piecewise_constant learning_rate_schedule
  # optimizer needs learning rate boundaries and learning rate values
  #learning_rate_decay learning_rate_values
  if hasattr(settings, 'learning_rate_schedule'):
    if settings.learning_rate_schedule == 'piecewise_constant':
      # set default if none provided
      if not (settings.learning_rate_decay or settings.learning_rate_values):
        settings.learning_rate_decay = 0.5

def _validate_settings(settings):
  # TODO: add more validations

  # assert settings.stride_system == 1 and settings.stride_network == 1, (
  #     'For now only stride of 1 is supported for stride_{system, network}.')

  assert all([settings.height_network == settings.height_feature_extractor,
              settings.width_network == settings.width_feature_extractor]), (
                  f"For now height_network ({settings.height_network}), "
                  f"height feature_extractor ({settings.height_feature_extractor}), "
                  f"and width_network ({settings.width_network}), "
                  f"width_feature_extractor ({settings.width_feature_extractor}) "
                  "should be equal.")

  # validate checkpoint paths only during training
  if hasattr(settings, 'init_ckpt_path'):
    if settings.init_ckpt_path:
      # log_dir must be empty of checkpoints
      assert not tf.train.latest_checkpoint(settings.log_dir), (
          'If init_ckpt_path is given log_dir must be empty of checkpoints, '
          'otherwise training continuous from latest checkpoint and '
          'init_ckpt_path has to be empty.')

  # learning rate related
  if hasattr(settings, 'learning_rate_schedule'):
    if settings.learning_rate_schedule == 'piecewise_constant':
      # NXOR of attributes (only one has to be given)
      if not (bool(settings.learning_rate_decay) != bool(settings.learning_rate_values)):
        raise AttributeError('If `learning_rate_schedule` is `piecewise_constant` exactly one of '
                             '`learning_rate_decay` or `learning_rate_values` must be given.')

  # problem def related
  lids2cids_unique = set(settings.training_problem_def['lids2cids'])
  cid_max = max(lids2cids_unique)
  # remove -1 if present
  lids2cids_unique.discard(-1)
  if not (lids2cids_unique == set(range(cid_max + 1))):
    raise ValueError('lids2cids field in training problem definition contains not continuous class ids.')
