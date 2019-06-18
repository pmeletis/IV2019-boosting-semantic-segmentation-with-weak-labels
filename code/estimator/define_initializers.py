import os

import tensorflow as tf
from tensorflow.contrib.distribute.python.values import DistributedValues


def train_init(config, params, scope='initializer'):
  # currently supported initialization:
  #   0) start training from scratch
  #   1) initialize from init_ckpt_path (log_dir has to be empty of checkpoints)
  #   2) continue training from log_dir

  del config

  # next two lines were added for distributed debugging
  if params.distribute:
    print('\n\ntrain_init returns None, None for distributed debugging.\n\n')
    return None, None

  with tf.name_scope(scope), tf.device('/cpu:0'):

    # assert bool(params.init_ckpt_path) != bool(tf.train.latest_checkpoint(params.log_dir)), (
    #     'If init_ckpt_path is given log_dir has to be empty of checkpoints, '
    #     'if log_dir is given training continuous from latest checkpoint and '
    #     'init_ckpt_path has to be empty.')

    ## initialize from checkpoint, e.g. trained on ImageNet
    # an empty string '' is False
    if params.init_ckpt_path:
      # the best we can do is to initialize from the checkpoint as much variables as possible
      # so we find the mapping from checkpoint names to model names
      # assumes names in model are extended with a prefix from names in checkpoint
      # e.g.
      # in checkpoint: resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
      # in model: feature_extractor/base/resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights

      # list of (name, shape) of checkpoint variables
      ckpt_vars = tf.train.list_variables(params.init_ckpt_path)
      # list of tf.Variable of model variables
      global_vars = tf.global_variables()

      # checkpoint variable name --> model variable mappings
      # TODO: exclude variables in a better way, still the parts below may be included in a
      # useful variable, e.g. use moving_average_variables and variables from models.py
      # TODO(panos): here it is assumed initialization from imagenet
      exclude = ['global_step', 'train_ops', 'ExponentialMovingAverage',
                 'Momentum', 'classifier', 'extension', 'psp', 'aspp']
      var_dict = dict()
      for gv in global_vars:
        for cvn, cvs in ckpt_vars:
          for exc in exclude:
            if exc in gv.name:
              break
          else:
            if cvn in gv.name and tf.TensorShape(cvs).is_compatible_with(gv.shape):
              var_dict[cvn] = gv

      extra_vars_to_init = set(global_vars).difference(set(var_dict.values()))

      init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
          params.init_ckpt_path,
          var_dict,
          ignore_missing_vars=False)
      extra_init_op = tf.variables_initializer(extra_vars_to_init)

      return tf.group(init_op, extra_init_op), init_feed_dict

    else:
      # start from scratch or continue from log_dir
      return None, None

    # previous_ckpt has priority over init_ckpt_path
    # if previous_ckpt is provided then it can be a directory or a previous checkpoint
    # and initialization is done accordingly
    # if it is not provided then ...
    # if params.previous_ckpt:
    #   # TODO: assuming previous_ckpt doesn't have only train_ops variables
    #   # collect variables that are not initialized from previous checkpoint
    #   excluded = ['train_ops']
    #   extra_vars_to_initialize = []
    #   for var in tf.global_variables():
    #     for exc in excluded:
    #       if exc in var.op.name:
    #         extra_vars_to_initialize.append(var)
    #   extra_init_op = tf.variables_initializer(extra_vars_to_initialize)
    #   #print('debug:extra_init_op:', extra_init_op)

    #   if os.path.isdir(params.previous_ckpt):
    #     # WARNING: by this line TF has created a new checkpoint, so the name
    #     #   returned is of the new checkpoint, e.g. model.ckpt-<old_step + 1>
    #     checkpoint_path = tf.train.latest_checkpoint(params.previous_ckpt)
    #   elif os.path.isfile(params.previous_ckpt + '.meta'):
    #     assert False, 'Resuming training from specific checkpoint is not supported yet by TF Estimator API.'
    #     #checkpoint_path = params.previous_ckpt
    #   else:
    #     assert False, 'Error in train_init.' #, possible both previous_ckpt and init_ckpt_path are None.'
    #   print('debug:checkpoint:', checkpoint_path)

    #   return extra_init_op, None

    # if params.init_ckpt_path:
    #   checkpoint_path = params.init_ckpt_path
    #   print('debug:checkpoint:', checkpoint_path)
    #   # don't restore vars containing those strings
    #   exclude = [get_unique_variable_by_name_without_creating('global_step').op.name,
    #              'ExponentialMovingAverage',
    #              'Momentum',
    #              'classifier',
    #              'extension']

    #   objects_to_restore = {}
    #   for var in get_saveable_objects_list(tf.get_default_graph()):
    #     for exc in exclude:
    #       if exc in var.op.name:
    #         break
    #     else:
    #       objects_to_restore[var.op.name[len('feature_extractor/base')+1:]] = var

    #   #for key in sorted(objects_to_restore):
    #   #  print('to restore:', key) #, objects_to_restore[key].op.name)

    #   init_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, objects_to_restore)

    #   # collect variables that are not initialized from imagenet model
    #   extra_vars_to_initialize = []
    #   for var in tf.global_variables():
    #     for exc in exclude:
    #       if exc in var.op.name:
    #         extra_vars_to_initialize.append(var)
    #   extra_init_op = tf.variables_initializer(extra_vars_to_initialize)
    #   init_op = tf.group(init_op, extra_init_op)

    #   return init_op, init_feed_dict

def replace_initializers(config, params, scope='replaced_initializers'):
  # currently supported initialization:
  #   0) start training from scratch
  #   1) initialize from init_ckpt_path (log_dir has to be empty of checkpoints)
  #   2) continue training from log_dir

  del config

  with tf.name_scope(scope), tf.device('/cpu:0'):
    ## initialize from checkpoint, e.g. trained on ImageNet
    # an empty string '' is False
    if params.init_ckpt_path:
      # the best we can do is to initialize from the checkpoint as much variables as possible
      # so we find the mapping from checkpoint names to model names
      # assumes names in model are extended with a prefix from names in checkpoint
      # e.g.
      # in checkpoint: resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
      # in model: feature_extractor/base/resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights

      # list of (name, shape) of checkpoint variables
      ckpt_vars = tf.train.list_variables(params.init_ckpt_path)
      # list of tf.Variable of model variables
      global_vars = tf.global_variables()

      # checkpoint variable name --> model variable mappings
      # TODO: exclude variables in a better way, still the parts below may be included in a
      # useful variable, e.g. use moving_average_variables and variables from models.py
      # TODO(panos): here it is assumed initialization from imagenet
      exclude = ['global_step', 'train_ops', 'ExponentialMovingAverage',
                 'Momentum', 'classifier', 'extension']
      if not params.psp_module:
        exclude.append('psp')
      # if not params.aspp_module:
      #   exclude.append('aspp')
      var_dict = dict()
      for gv in global_vars:
        for cvn, cvs in ckpt_vars:
          for exc in exclude:
            if exc in gv.name:
              break
          else:
            if cvn in gv.name and tf.TensorShape(cvs).is_compatible_with(gv.shape):
              var_dict[cvn] = gv

      # extra_vars_to_init = set(global_vars).difference(set(var_dict.values()))

      # for now init_from_checkpoint doesn't support DistributedValues (TF bug, error)
      # so do a scan and unwrap DistibutedValues
      for k, v in var_dict.items():
        if isinstance(v, DistributedValues):
          var_dict[k] = v.get()
        else:
          # keep default behavior
          pass
      # suppress INFO logging messages
      with _temp_verbosity(tf.logging.WARN):
        tf.train.init_from_checkpoint(params.init_ckpt_path, var_dict)
    else:
      # start from scratch or continue from log_dir (managed by estimator)
      pass

class _temp_verbosity():
  def __init__(self, temp_verbosity):
    self._previous_verbosity = tf.logging.get_verbosity()
    self._temp_verbosity = temp_verbosity

  def __enter__(self):
    tf.logging.set_verbosity(self._temp_verbosity)
    return self._temp_verbosity

  def __exit__(self, *args):
    tf.logging.set_verbosity(self._previous_verbosity)
