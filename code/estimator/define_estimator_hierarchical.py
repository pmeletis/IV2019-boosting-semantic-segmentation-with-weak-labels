"""Definition of TF Estimator.
"""

import numpy as np
import copy
import os
import itertools
from operator import itemgetter

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import timeline
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.training import create_train_op
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
resnet_arg_scope = resnet_utils.resnet_arg_scope
from tensorflow.contrib.distribute.python.values import (
    TowerLocalVariable, MirroredVariable, DistributedValues)
# in TF 1.1 metrics_impl has _streaming_confusion_matrix hidden method
from tensorflow.python.ops import metrics_impl

from estimator.define_losses_hierarchical import define_losses
from estimator.define_savers import train_saver, predict_saver, evaluate_saver
from estimator.define_optimizer import define_optimizer
from estimator.define_initializers import train_init
from estimator.define_metrics import mean_iou
from estimator.define_initializers import replace_initializers
from utils.utils import (almost_equal, get_unique_tensor_by_name_without_creating,
                         print_tensor_info, get_unique_variable_by_name_without_creating,
                         get_saveable_objects_list, _replacevoids)
from input_pipelines.utils import get_temp_Nb

_ALLOWED_MODES = {tf.estimator.ModeKeys.TRAIN,
                  tf.estimator.ModeKeys.EVAL,
                  tf.estimator.ModeKeys.PREDICT}

def define_estimator(mode, features, labels, model_fn, config, params):
  """Add documentation... More information at tf.Estimator class.
  Assumptions:
    features: a dict containing rawfeatures and profeatures
      both: Nb x hf x wf x 3, tf.float32, in [0,1]
    labels: a dict containing rawfeatures and profeatures
      both: Nb x hf x wf, tf.int32, in [0,Nc-1]
  Args:
    features: First item returned by input_fn passed to train, evaluate, and predict.
    labels: Second item returned by input_fn passed to train, evaluate, and predict.
    mode: one of tf.estimator.ModeKeys.
    config: a tf.estimator.RunConfig object...
    parameters: a tf.train.HParams object...
    ...
  """

  ## arguments assertions: make sure that everything used by this function
  ##   is provided correctly
  # arguments assertions
  # assert mode and config and params, (
  #   'For now mode, config and params must be provided.')
  assert mode in _ALLOWED_MODES, (
      'mode should be TRAIN, EVAL or PREDICT from tf.estimator.ModeKeys.')
  # assert features and labels, (
  #   'Basic assertion for features and labels. '
  #   'To be elaborated later, so errors are thrown from this point for safety.')
  # attrs = ['model', 'sfe', 'batch_norm_accumulate_statistics',
  #          'projection_dims', 'classifier_rate',
  #          'classifier_kernel', 'config', 'upsample']
  # for attr in attrs:
  #   assert hasattr(params, attr), 'params must have ' + attr + 'attribute.'
  # model assertions
  assert params.name_feature_extractor in {'resnet_v1_50', 'resnet_v1_101'}, (
      'params must have name_feature_extractor attribute in resnet_v1_{50,101}.')
  if params.name_feature_extractor == 'resnet_v1_101':
    raise NotImplementedError(
        'Use of resnet_v1_101 as base feature extractor is not yet implemented.')

  # unpack features
  rawimages = features['rawimages'] if 'rawimages' in features.keys() else None
  rawimagespaths = features['rawimagespaths'] if 'rawimagespaths' in features.keys() else None
  proimages = features['proimages']
  prolabels = labels if labels else None

  # print('debug:rawimages, proimages, prolabels:', rawimages, proimages, prolabels)

  ## build a fully convolutional model for semantic segmentation
  # predictions refer to the training class ids
  # for plotting of results (inference) or assessment, predictions should be transformed
  #   using `{inference, evaluation}_problem_def`s
  _, _, predictions = model_fn(mode, proimages, prolabels, config, params)

  # TODO(panos): assert that proimages and predictions have same spatial size

  if mode == tf.estimator.ModeKeys.TRAIN:

    # global step
    global_step = tf.train.get_or_create_global_step()

    # losses
    with tf.variable_scope('losses'):
      losses = define_losses(mode, predictions, prolabels, config, params)

    # exponential moving averages
    # creates variables in checkpoint with name: 'emas/' + <variable_name> +
    #   {'ExponentialMovingAverage,Momentum}
    # ex.: for 'classifier/logits/Conv/biases' it saves also
    #          'emas/classifier/logits/Conv/biases/ExponentialMovingAverage'
    #      and 'emas/classifier/logits/Conv/biases/Momentum'
    # create_train_op guarantees to run GraphKeys.UPDATE_OPS collection
    #   before total_loss in every step, but doesn't give any guarantee
    #   for running after some other op, and since ema need to be run
    #   after applying the gradients maybe this code needs checking
    # TODO(panos): variables should be in the MODEL_VARIABLES collection in order to
    #   be taken with emas
    # TODO(panos): investigate: in the distributed setting mirrored variables are not
    #   saved as extra variables but are computed per device and then reduced
    #   so practically emas should only be saved for one set of per device variables
    #   since broadcasting updates them in every step, here we chose /gpu:0 since
    #   it is most of the times the empty device (no linux processes on it)
    # next line for distributed debugging
    # distribution_strategy = tf.contrib.distribute.get_tower_context()
    # TODO(panos): assuming that /gpu:0 is always used and is the "empty" gpu
    # TODO(panos): find out why in a distribution.scope() tf.device doesn't work
    #   and this code needs to put variables in a specific tower locality
    # running_on_gpu0 = 'gpu:0' in  tf.contrib.distribute.get_tower_context().device.lower()
    if params.ema_decay > 0:
      with tf.variable_scope('exponential_moving_averages'):
        #for mv in slim.get_model_variables():
        #  print('slim.model_vars:', mv.op.name)
        ema = tf.train.ExponentialMovingAverage(params.ema_decay,
                                                num_updates=global_step,
                                                zero_debias=True)
        variables_to_ema = []
        for mv in tf.model_variables():
          if 'BatchNorm/moving' not in mv.name:
            variables_to_ema.append(mv)
        print(
            f"\nFound {len(tf.model_variables())} variables, saving exponential "
            f"moving averages for {len(variables_to_ema)} of them.\n")
        maintain_ema_op = ema.apply(var_list=variables_to_ema)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_ema_op)

    # create training operation
    with tf.variable_scope('train_ops'):

      # optimizer
      optimizer = define_optimizer(global_step, params)

      # training op
      train_op = create_train_op(
          losses['total'],
          optimizer,
          global_step=global_step,
          # update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
          summarize_gradients=False,
          # transform_grads_fn=,
          # gradient_multipliers=gradient_multipliers,
          check_numerics=False,
      )

    # TODO: maybe parameterize it
    training_hooks = [
        _RunMetadataHook(params.log_dir,
                         every_n_iter=max(params.num_training_steps//50,
                                          params.save_checkpoints_steps))]

    # next two lines were added for distributed debugging
    if params.distribute:
      tower_context = tf.contrib.distribute.get_tower_context()
      assert tower_context
      print(f"Tower {tower_context.tower_id}: _RunMetadataHook is not supported "
            "yet for distributed training.")
      training_hooks = []

    # for gv in tf.global_variables():
    #   if isinstance(gv, TowerLocalVariable):
    #     print('T', end='')
    #   elif isinstance(gv, MirroredVariable):
    #     print('M', end='')
    #   else:
    #     NotImplementedError()
    #   for d, v in gv._index.items():
    #     print('  ', d, v.op.name)

    replace_initializers(config, params)

    summaries_data = {'features': features,
                      'labels': labels,
                      'predictions': predictions,
                      'losses': losses,
                      'learning_rate': optimizer._learning_rate} #pylint: disable=protected-access

    scaffold = _define_scaffold(mode, config, params, summaries_data)
    estimator_spec = tf.estimator.EstimatorSpec(mode,
                                                predictions=predictions,
                                                loss=losses['total'],
                                                train_op=train_op,
                                                training_hooks=training_hooks,
                                                scaffold=scaffold)

  if mode == tf.estimator.ModeKeys.EVAL:
    with tf.variable_scope('losses'):
      losses = define_losses(mode, predictions, prolabels, config, params)

    # returns (variable, update_op)
    # TF internal error/problem: _streaming_confusion_matrix internally casts
    # labels and predictions to int64, and since we feed a dictionary, tensors are
    # passed by reference leading them to change type, thus we send an identity
    # confusion_matrix = metrics_impl._streaming_confusion_matrix(  # pylint: disable=protected-access
    #     tf.identity(prolabels),
    #     tf.identity(predictions['decisions']),
    #     params.output_Nclasses)
    # l1_probs, decs = itemgetter('l1_probabilities', 'decisions')(predictions)
    # create a new dict with the supported keys only
    predictions = _map_predictions_to_new_cids(predictions, params.training_cids2evaluation_cids)
    if params.replace_voids:
      predictions = _replace_voids(predictions, params)
    # TODO(panos): confusion matrix expects prolabels and predictions to have the same shape
    #   this may not the case when preserve_aspect_ratio is set and this will give an error
    if hasattr(params, 'preserve_aspect_ratio'):
      if params.preserve_aspect_ratio:
        raise NotImplementedError('evaluation with preserving aspect ratio is not implemented.')
    predictions = _resize_predictions(predictions, tf.shape(labels['prolabels'])[1:3], params)
    tcids2ecids = _replacevoids(params.training_cids2evaluation_cids)
    confusion_matrix = metrics_impl._streaming_confusion_matrix(  # pylint: disable=protected-access
        labels['prolabels'],
        predictions['decisions'],
        # +1 due to convention of starting counting at 0
        max(tcids2ecids) + 1)

    # dict of metrics keyed by name with values tuples of (metric_tensor, update_op)
    # TODO: add more semantic segmentation metrics
    eval_metric_ops = {'confusion_matrix': (
        tf.to_int32(confusion_matrix[0]), confusion_matrix[1])}

    scaffold = _define_scaffold(mode, config, params)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=losses['total'],
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # create a new dict with the supported keys only
    l1_probs, l2_vehicle_probs, l2_human_probs, decs = itemgetter(
        'l1_probabilities', 'l2_vehicle_probabilities', 'l2_human_probabilities', 'decisions')(
            predictions)
    predictions = {'l1_probabilities': l1_probs,
                   'l2_vehicle_probabilities': l2_vehicle_probs,
                   'l2_human_probabilities': l2_human_probs,
                   'decisions': decs}
    # workaround for connecting input pipeline outputs to system output
    # TODO(panos): maybe from a system perspective makes more sense to have mapping and
    #   resizing in the system_factory
    # since these are functions of the system and not the network/estimator
    # new size defaults to provided values
    # if at least one is None then new size is the arbitrary size of rawimage in each step
    new_size = (params.height_system, params.width_system)
    is_arbitrary = not all(new_size)
    if is_arbitrary:
      if rawimages is not None:
        predictions['rawimages'] = rawimages
      if rawimagespaths is not None:
        predictions['rawimagespaths'] = rawimagespaths
      new_size = tf.shape(predictions['rawimages'])[1:3]
    predictions = _resize_predictions(predictions, new_size, params)
    tf.logging.warn('Mapping of predictions to new cids is not implemented for now.')
    # predictions = _map_predictions_to_new_cids(predictions, params.training_cids2inference_cids)
    if params.replace_voids:
      predictions = _replace_voids(predictions, params)

    scaffold = _define_scaffold(mode, config, params)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        scaffold=scaffold)

  return estimator_spec

def _define_scaffold(mode, config, params, summaries_data=None):
  """Creates scaffold containing initializers, savers and summaries.

  Args:
    summaries_data: dictionary containing all tensors needed for summaries during training

  Returns:
    a tf.train.Scaffold instance
  """
  # Comment: init_op with init_feed_dict, and init_fn are executed from SessionManager
  # only if model is not loaded successfully from checkpoint using the saver.
  # if no saver is provided then the default saver is constructed to load all
  # variables (from collections GLOBAL_VARIABLES and SAVEABLE_OBJECTS) and init_op won't
  # be executed.
  # For that reason, during training using init_checkpoint we provide a custom saver only
  # for model variables and an init_op to initialize all variables not in init_checkpoint.

  # create scopes outside of scaffold namescope
  # with tf.name_scope('init') as init_scope:
  #   pass
  with tf.name_scope('saver') as saver_scope:
    pass

  with tf.name_scope('scaffold'):
    if mode == tf.estimator.ModeKeys.TRAIN:
      _define_summaries(mode, config, params, summaries_data)
      saver = train_saver(config, params, scope=saver_scope)
      # Initialization is handled by replace_initializers
      # init_op, init_feed_dict = train_init(config, params, scope=init_scope)
      # init_op, init_feed_dict = [None]*2
    elif mode == tf.estimator.ModeKeys.EVAL:
      saver = evaluate_saver(config, params, scope=saver_scope)
      # init_op, init_feed_dict = [None]*2
    elif mode == tf.estimator.ModeKeys.PREDICT:
      saver = predict_saver(config, params, scope=saver_scope)
      # init_op, init_feed_dict = [None]*2

    # WARNING: default ready_op and ready_for_local_init_op install operations
    #   in the graph to report_uninitialized_variables, resulting in too many ops,
    #   so make ready_for_local_init_op a no_op to reduce them.
    scaffold = tf.train.Scaffold(
        # init_op=init_op,
        # init_feed_dict=init_feed_dict,
        # ready op only for distributed debugging
        # ready_op=tf.no_op(),
        saver=saver)

  return scaffold

def _define_summaries(mode, config, params, summaries_data):
  # this function is only to be used for training mode

  assert mode == tf.estimator.ModeKeys.TRAIN, print('internal error: summaries only for training.')

  with tf.name_scope('summaries'), tf.device('/cpu:0'):
    # unpack necessary objects and tensors
    # WARNING: assumes all necessary items exist (maybe add assertions)
    # rawlabels = summaries_data['labels']['rawlabels']
    proimages = summaries_data['features']['proimages']
    prolabels_per_pixel = summaries_data['labels']['prolabels_per_pixel']
    prolabels_per_bbox = summaries_data['labels']['prolabels_per_bbox']
    l1_probs, l1_decs, l2_vehicle_probs, l2_vehicle_decs, l2_human_probs, l2_human_decs, decs = itemgetter(
        'l1_probabilities', 'l1_decisions', 'l2_vehicle_probabilities',
        'l2_vehicle_decisions', 'l2_human_probabilities',
        'l2_human_decisions', 'decisions')(
            summaries_data['predictions'])
    # create a new dict with the supported keys only
    # predictions = _map_predictions_to_new_cids(
        # {'probabilities': probs, 'decisions': decs}, params.training_cids2inference_cids)
    # probs, decs = itemgetter('probabilities', 'decisions')(predictions)
    tot_loss, reg_loss, l1_seg_loss, l1_seg_loss_hot, l2_vehicle_seg_loss, l2_human_seg_loss = itemgetter(
        'total', 'regularization', 'l1_segmentation', 'l1_segmentation_hot',
        'l2_vehicle_segmentation', 'l2_human_segmentation')(
            summaries_data['losses'])

    # drawing
    with tf.name_scope('drawing'):
      with tf.name_scope('palette'):
        palette = tf.constant(params.training_problem_def['cids2colors'], dtype=tf.uint8)

      # WARNING: assuming upsampling, that is all color_* images have the
      # same spatial dimensions
      if params.per_pixel_dataset_name == 'vistas':
        # human: 19->19, vehicle: 49->52
        l1_cids2common_cids = tf.cast([
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
          33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
          63, 64, 65], tf.int32)
        per_bbox_cids2common_cids = tf.cast([52, 54, 55, 57, 58, 61, 19, 19, 19, 19, 19, 48, 50, 50, 65], tf.int32)
      elif params.per_pixel_dataset_name == 'cityscapes':
        l1_cids2common_cids = tf.cast([
           0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          10,11,13,19], tf.int32)
        per_bbox_cids2common_cids = tf.cast([18, 15, 13, 17, 16, 14, 11, 11, 11, 11, 11, 6, 7, 7, 19], tf.int32)
      color_l1_decisions = _cids2col(tf.gather(l1_cids2common_cids, l1_decs), palette)
      color_l2_vehicle_decisions = _cids2col(l2_vehicle_decs, palette)
      color_decisions = _cids2col(decs, palette)

      # generate confidence image, preventing TF from normalizing max prob
      # to 1, by casting to tf.uint8
      color_l1_confidences = tf.stack(
          [tf.cast(tf.reduce_max(l1_probs, axis=3)*255, tf.uint8)]*3, axis=3)
      # raise to ^50 for more contrast in high probabilities
      color_l2_vehicle_confidences = tf.stack(
          [tf.cast(tf.reduce_max(tf.pow(l2_vehicle_probs, 50), axis=3)*255, tf.uint8)]*3,
          axis=3)

      color_prolabels = _cids2col(
          tf.concat(
              [prolabels_per_pixel,
               tf.gather(per_bbox_cids2common_cids,
                         tf.argmax(prolabels_per_bbox, axis=-1, output_type=tf.int32))],
              0),
          palette)

      # TODO(panos): as noted in MirrorStrategy, in a multi-gpu setting the effective
      #   batch size is num_gpus * Nb, however in the early implementation
      #   (master branch of April 1st 2018), summaries are computed per GPU
      #   and since Nb >= Nb/num_gpus (of the future implementation)
      #   no change is needed here
      # TODO(panos): here it is assumed that the input pipeline outputs proimages in [-1, 1)
      tf.summary.image('proimages',
                       (proimages + 1) / 2,
                       # tf.image.convert_image_dtype(proimages, tf.uint8, saturate=True),
                       max_outputs=100,
                       family='preprocessed_data')
      tf.summary.image('prolabels',
                       color_prolabels,
                       max_outputs=100,
                       family='preprocessed_data')
      tf.summary.image('decisions', color_decisions, max_outputs=100, family='results')
      tf.summary.image('l1_decisions', color_l1_decisions, max_outputs=100, family='results')
      tf.summary.image('l2_vehicle_decisions', color_l2_vehicle_decisions, max_outputs=100, family='results')
      tf.summary.image('l1_confidences', color_l1_confidences, max_outputs=100, family='results')
      tf.summary.image('l2_vehicle_confidences_stretched', color_l2_vehicle_confidences, max_outputs=100, family='results')

      # compute batch metrics
      m_iou_per_pixel = mean_iou(
          prolabels_per_pixel,
          decs[:get_temp_Nb(config, params.Nb_per_pixel), ...],
          num_classes=params.output_Nclasses,
          params=params)

    # TODO: in order to disable loss summary created internally by estimator this line should
    # evaluate to False:
    # not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)])
    tf.summary.scalar('total', tot_loss, family='losses')
    tf.summary.scalar('regularization', reg_loss, family='losses')
    tf.summary.scalar('l1_segmentation', l1_seg_loss, family='losses')
    # tf.summary.scalar('l1_segmentation_hot', l1_seg_loss_hot, family='losses')
    tf.summary.scalar('l2_vehicle_segmentation', l2_vehicle_seg_loss, family='losses')
    tf.summary.scalar('l2_human_segmentation', l2_human_seg_loss, family='losses')
    tf.summary.scalar('mIoU', m_iou_per_pixel, family='metrics')

    tf.summary.scalar('learning_rate', summaries_data['learning_rate'], family='optimizer')

    #variables_to_summarize = slim.get_variables()
    #for var in variables_to_summarize:
      #if ('classifier' in var.op.name) and ('Momentum' not in var.op.name):
        ##tf.summary.histogram(var.op.name + '/summary', var)
        #pass
      #else:
        #pass

    # misc summaries
    # ups_filter is a 4D Tensor: 16x16x1x1
    #ups_filter = get_unique_variable_by_name_without_creating('classifier/upsampling_shared/weights')
    #if ups_filter:
      #tf.summary.image('ups_filter/kernel', ups_filter.value()[tf.newaxis,...,0])
      #tf.summary.scalar('ups_filter/bias', get_unique_variable_by_name_without_creating('classifier/upsampling_shared/biases').value()[0])

def _cids2col(cids, palette):
  # cids: Nb x H x W, tf.int32, with class ids in [0,Nc-1]
  # palette: Nc x 3, tf.uint8, with rgb colors in [0,255]
  # returns: Nb x H x W x 3, tf.uint8, in [0,255]

  # TODO: add type checking
  return tf.gather_nd(palette, tf.expand_dims(cids, axis=-1))

class _RunMetadataHook(tf.train.SessionRunHook):
  """Exports the run metadata as a trace to log_dir every N local steps or every N seconds.
  """
  # TODO: implement this with tf.profiler

  def __init__(self, log_dir, every_n_iter=None, every_n_secs=None):
    """Initializes a `_RunMetadataHook`.

    Args:
      log_dir: the log_dir directory to save traces.
      every_n_iter: `int`, save traces once every N local steps.
      every_n_secs: `int` or `float`, save traces once every N seconds.

      Exactly one of `every_n_iter` and `every_n_secs` should be provided.

    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError("Exactly one of every_n_iter and every_n_secs must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError(f"Invalid every_n_iter={every_n_iter}.")
    self._timer = tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter)
    self._iter_count = None
    self._should_trigger = None
    self._tf_global_step = None
    self._np_global_step = None
    self._log_dir = log_dir

  def begin(self):
    self._timer.reset()
    self._iter_count = 0

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    self._tf_global_step = tf.train.get_global_step()
    # at this moment graph is finalized and get_global_step cannot create an identity
    #   read operation, thus get the value through the read op of the variable
    # self._tf_global_step = tf.train.get_global_step().value
    assert self._tf_global_step, 'Internal error: _RunMetadataHook cannot retrieve global step.'

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      self._timer.update_last_triggered_step(self._iter_count)
      return tf.train.SessionRunArgs(
          fetches=self._tf_global_step,
          options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
    else:
      return None

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if self._should_trigger:
      self._np_global_step = run_values.results
      # self._iter_count = self._np_global_step
      self._timer.update_last_triggered_step(self._iter_count)
      run_metadata = run_values.run_metadata
      if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(self._log_dir, f"tf_trace-{self._np_global_step}.json")
        tf.logging.info(f"Writing trace to {trace_filename}.")
        file_io.write_string_to_file(trace_filename, trace)
        # TODO: add run_metadata to summaries with summary_writer
        #   find how summaries are saved in the estimator and add them
        # summary_writer.add_run_metadata(run_metadata, f"run_metadata-{self._global_step}")

    self._iter_count += 1

def _have_compatible_shapes(lot):
  # lot: list_of_tensors
  tv = True
  for t1, t2 in itertools.combinations(lot, 2):
    tv = tv and t1.shape.is_compatible_with(t2.shape)
  return tv

def _have_equal_shapes(lot):
  # lot: list_of_tensors
  tv = True
  for t1, t2 in itertools.combinations(lot, 2):
    tv = tv and (t1.shape == t2.shape)
  return tv

def _map_predictions_to_new_cids(predictions, old_cids2new_cids):
  """Map training predictions to predictions according to inference problem definition."""
  # transform predictions with new class ids: decs and probs correspond to training class ids
  # for plotting they should be transformed to new (inference or evaluation) class ids
  #   e.g. output_Nclasses = 5,
  #        training_cids2inference_cids = [-1, 1, 1, 0, -1] --> [2, 1, 1, 0, 2]
  #        before: probs.shape: (..., 5) --> after: probs.shape: (..., 3)
  # for probs: using the probability of union rule: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
  #   with P(A ∩ B) = 0 probabilities have to be summed
  # TODO(panos): save computations by checking if inference and training
  #   problem defs are the same

  supported_keys = {
      'l1_probabilities', 'rawimages', 'rawimagespaths', 'decisions', 'l1_decisions',
      'l1_logits', 'l1_probabilities', 'l2_vehicle_decisions', 'l2_vehicle_logits',
      'l2_vehicle_probabilities', 'l2_human_decisions', 'l2_human_logits',
      'l2_human_probabilities'}
  # check if supported_keys is a superset of predictions.keys()
  assert supported_keys >= predictions.keys(), (
      f"Supported keys are {sorted(supported_keys)} and predictions keys are {sorted(set(predictions.keys()))}. "
      f"Supported keys must be a superset of predictions keys for resizing predictions.")

  probs, decs = itemgetter('l1_probabilities', 'decisions')(predictions)
  old_cids2new_cids = _replacevoids(old_cids2new_cids)
  ocids2ncids = tf.cast(old_cids2new_cids, tf.int32)
  decs = tf.gather(ocids2ncids, decs)
  try:
    probs_transposed = tf.transpose(probs, (3, 0, 1, 2))
    probs_transformed = tf.unsorted_segment_sum(
        probs_transposed, ocids2ncids, tf.reduce_max(ocids2ncids) + 1)
    probs = tf.transpose(probs_transformed, (1, 2, 3, 0))
  except:
    tf.logging.info('\nl1_probabilities are not transformed to new cids.\n')
  tf.logging.info('\nOnly decisions were transformed to new cids.\n')

  new_predictions = predictions
  new_predictions.update({'l1_probabilities': probs, 'decisions': decs})

  return new_predictions

def _resize_predictions(predictions, new_size, settings): #pylint: disable=unused-argument
  # predictions: dictionary of tf.Tensor
  # resize predictions to size
  # size: tf.int32, (?, ?)
  # TODO(panos): this function is meant to replace _resize_predictions
  # only `decisions` and `probabilities` are suppported for now

  # this function "knows" how to resize only the following supported keys
  supported_keys = {
      'l1_probabilities', 'rawimages', 'rawimagespaths', 'decisions', 'l1_decisions',
      'l1_logits', 'l1_probabilities', 'l2_vehicle_decisions', 'l2_vehicle_logits',
      'l2_vehicle_probabilities', 'l2_human_decisions', 'l2_human_logits',
      'l2_human_probabilities'}
  # check if supported_keys is a superset of predictions.keys()
  assert supported_keys >= predictions.keys(), (
      f"Supported keys are {sorted(supported_keys)} and predictions keys are "
      f"{sorted(set(predictions.keys()))}. Supported keys must be a superset of "
      "predictions keys for resizing predictions.")

  old_decs = predictions['decisions']
  old_l1_probs = predictions['l1_probabilities']
  old_l2_vehicle_probs = predictions['l2_vehicle_probabilities']
  old_l2_human_probs = predictions['l2_human_probabilities']

  # TODO(panos): save computation by comparing size
  # resize decisions and probabilities
  new_l1_probs = tf.image.resize_images(old_l1_probs, new_size, align_corners=True)
  new_l2_vehicle_probs = tf.image.resize_images(old_l2_vehicle_probs, new_size, align_corners=True)
  new_l2_human_probs = tf.image.resize_images(old_l2_human_probs, new_size, align_corners=True)
  new_decs = tf.image.resize_images(
      old_decs[..., tf.newaxis],
      new_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)[..., 0]

  new_predictions = predictions
  new_predictions.update({'l1_probabilities': new_l1_probs,
                          'l2_vehicle_probabilities': new_l2_vehicle_probs,
                          'l2_human_probabilities': new_l2_human_probs,
                          'decisions': new_decs})

  return new_predictions

def _replace_voids(predictions, params):
  """
  Replace void pixel decisions with the next most probable (not ignored) class.
  The model can output void pixel decisions due to the following reasons:
    during training, if unlabeled pixels exist in ground truth (corresponding to label "-1"
      in lids2cids field in training problem definition) predictions may include a void label.
    during inference/evaluation, some labels are ignored (corresponding to label "-1" in
      training_cids2inference/evaluation_cids field in evaluation problem definition),
      thus predictions may have void labels.

  This function will replace the decisions key in the predictions dict to contain non-void
  and not ignored classes.
  """

  #TODO(panos): generalize this function to a generalized decision rule function

  supported_keys = {
      'l1_probabilities', 'rawimages', 'rawimagespaths', 'decisions', 'l1_decisions',
      'l1_logits', 'l1_probabilities', 'l2_vehicle_decisions', 'l2_vehicle_logits',
      'l2_vehicle_probabilities'}
  # check if supported_keys is a superset of predictions.keys()
  assert supported_keys >= predictions.keys(), (
      f"Supported keys are {supported_keys} and predictions keys are {predictions.keys()}. "
      f"Supported keys must be a superset of predictions keys for resizing predictions.")

  probs, decs = itemgetter('l1_probabilities', 'decisions')(predictions)
  # values: tf.float32, Nb x hf x wf x Nc
  # indices: tf.int32, Nb x hf x wf x Nc
  # TODO(panos): limit k to the maximum number of evaluated classes for gaining speed
  # top_k is much faster on CPU than GPU at least till TF v1.6
  with tf.device('/cpu:0'):
    # if hasattr(params, 'training_cids2inference_cids'):
    #   tcids2ncids = params.training_cids2inference_cids
    # elif hasattr(params, 'training_cids2evaluation_cids'):
    #   tcids2ncids = params.training_cids2evaluation_cids
    # else:
    #   assert False, 'Code shoudn\'t reach this point. internal error.'
    # voids_in_tcids2ncids = -1 in tcids2ncids
    # previous mapping and aggregation of probabilities makes only one channel for void
    values, indices = tf.nn.top_k(probs, k=2) # 4-D, 4-D
    # sanity check
    asserions = [tf.equal(decs, indices[..., 0])]
    with tf.control_dependencies(asserions):
      indices = tf.identity(indices)

  void_mask = tf.equal(decs, tf.shape(probs)[-1] - 1)
  new_decs = tf.where(void_mask,
                      indices[..., 1],
                      indices[..., 0])
  # TODO(panos): translate also probs
  tf.logging.warn(
      'probabilities were not mapped to the replaced voids in decisions, '
      'thus they represent the probabilities before replacing.')

  new_predictions = predictions
  new_predictions.update({'l1_probabilities': probs, 'decisions': new_decs})

  return new_predictions
