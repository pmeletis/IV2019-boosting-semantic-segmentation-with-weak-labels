"""
Cross-replica (global) batch normalization.
"""

# from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras import constraints
# from tensorflow.python.keras import initializers
# from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
# from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
# from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribution_strategy_context
# from tensorflow.python.util.tf_export import tf_export
from tensorflow.contrib.layers.python.layers.layers import _build_variable_getter, _add_variable_to_collections
from tensorflow.python.ops import variable_scope
# from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils

import tensorflow as tf

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'

class CrossReplicaBatchNormalization(tf.layers.BatchNormalization):
  """
  First implementation tries to achieve approximate global batch normalization
  by changing the VariableAggregation of moving_{mean, variance} from ON_READ to ON_WRITE.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  # override the build method so we have control over the creation of moving_{mean, variance} variables
  def _not_used_build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    if not isinstance(self.axis, list):
      raise TypeError('axis must be int or list, type given: %s'
                      % type(self.axis))

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)

    if self.virtual_batch_size is not None:
      if self.virtual_batch_size <= 0:
        raise ValueError('virtual_batch_size must be a positive integer that '
                         'divides the true batch size of the input Tensor')
      # If using virtual batches, the first dimension must be the batch
      # dimension and cannot be the batch norm axis
      if 0 in self.axis:
        raise ValueError('When using virtual_batch_size, the batch dimension '
                         'must be 0 and thus axis cannot include 0')
      if self.adjustment is not None:
        raise ValueError('When using virtual_batch_size, adjustment cannot '
                         'be specified')

    if self.fused:
      # Currently fused batch norm doesn't support renorm. It also only supports
      # an input tensor of rank 4 and a channel dimension on axis 1 or 3.
      # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
      # output back to its original shape accordingly.
      self.fused = (not self.renorm and
                    ndims == 4 and
                    self.axis in [[1], [3]] and
                    self.virtual_batch_size is None and
                    self.adjustment is None)
      # TODO(chrisying): fused batch norm is currently not supported for
      # multi-axis batch norm and by extension virtual batches. In some cases,
      # it might be possible to use fused batch norm but would require reshaping
      # the Tensor to 4D with the axis in 1 or 3 (preferred 1) which is
      # particularly tricky. A compromise might be to just support the most
      # common use case (turning 5D w/ virtual batch to NCHW)

    if self.fused:
      if self.axis == [1]:
        self._data_format = 'NCHW'
      elif self.axis == [3]:
        self._data_format = 'NHWC'
      else:
        raise ValueError('Unsupported axis, fused batch norm only supports '
                         'axis == [1] or axis == [3]')

    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
      param_dtype = dtypes.float32
    else:
      param_dtype = self.dtype or dtypes.float32

    axis_to_dim = {x: input_shape[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                         input_shape)
    self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

    if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
      # Single axis batch norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [axis_to_dim[i] if i in axis_to_dim
                     else 1 for i in range(ndims)]
      if self.virtual_batch_size is not None:
        # When using virtual batches, add an extra dim at index 1
        param_shape.insert(1, 1)
        for idx, x in enumerate(self.axis):
          self.axis[idx] = x + 1      # Account for added dimension

    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True)
    else:
      self.gamma = None
      if self.fused:
        self._gamma_const = array_ops.constant(
            1.0, dtype=param_dtype, shape=param_shape)

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True)
    else:
      self.beta = None
      if self.fused:
        self._beta_const = array_ops.constant(
            0.0, dtype=param_dtype, shape=param_shape)

    try:
      # Disable variable partitioning when creating the moving mean and variance
      if hasattr(self, '_scope') and self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None
      self.moving_mean = self.add_weight(
          name='moving_mean',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.moving_mean_initializer,
          synchronization=tf_variables.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN)

      self.moving_variance = self.add_weight(
          name='moving_variance',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.moving_variance_initializer,
          synchronization=tf_variables.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN)

      if self.renorm:
        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = self.add_weight(
              name=name,
              shape=shape,
              dtype=param_dtype,
              initializer=init_ops.zeros_initializer(),
              synchronization=tf_variables.VariableSynchronization.ON_READ,
              trainable=False,
              aggregation=tf_variables.VariableAggregation.MEAN)
          return var

        with distribution_strategy_context.get_distribution_strategy(
        ).colocate_vars_with(self.moving_mean):
          self.renorm_mean = _renorm_variable('renorm_mean', param_shape)
          self.renorm_mean_weight = _renorm_variable('renorm_mean_weight', ())
        # We initialize renorm_stddev to 0, and maintain the (0-initialized)
        # renorm_stddev_weight. This allows us to (1) mix the average
        # stddev with the minibatch stddev early in training, and (2) compute
        # the unbiased average stddev by dividing renorm_stddev by the weight.
        with distribution_strategy_context.get_distribution_strategy(
        ).colocate_vars_with(self.moving_variance):
          self.renorm_stddev = _renorm_variable('renorm_stddev', param_shape)
          self.renorm_stddev_weight = _renorm_variable('renorm_stddev_weight',
                                                       ())
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  # override the call method so we have control over the use of moving_{mean, variance} variables
  def call(self, inputs, training=None):
    original_training_value = training
    if training is None:
      training = K.learning_phase()

    in_eager_mode = context.executing_eagerly()
    if self.virtual_batch_size is not None:
      # Virtual batches (aka ghost batches) can be simulated by reshaping the
      # Tensor and reusing the existing batch norm implementation
      original_shape = [-1] + inputs.shape.as_list()[1:]
      expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

      # Will cause errors if virtual_batch_size does not divide the batch size
      inputs = array_ops.reshape(inputs, expanded_shape)

      def undo_virtual_batching(outputs):
        outputs = array_ops.reshape(outputs, original_shape)
        return outputs

    if self.fused:
      outputs = self._fused_batch_norm(inputs, training=training)
      if self.virtual_batch_size is not None:
        # Currently never reaches here since fused_batch_norm does not support
        # virtual batching
        outputs = undo_virtual_batching(outputs)
      if not context.executing_eagerly() and original_training_value is None:
        outputs._uses_learning_phase = True  # pylint: disable=protected-access
      return outputs

    # Panos: make sure fused is true
    assert self.fused, 'Cross-replica batch norm is implemented only for fused for now.'

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.get_shape()
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    if training_value is not False:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: array_ops.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: moving_mean)
      variance = tf_utils.smart_cond(training,
                                     lambda: variance,
                                     lambda: moving_variance)

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
        new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
      else:
        new_mean, new_variance = mean, variance

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        if in_eager_mode and not self.trainable:
          return

        return self._assign_moving_average(var, value, self.momentum)

      mean_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, new_mean),
          lambda: self.moving_mean)
      variance_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, new_variance),
          lambda: self.moving_variance)
      if not context.executing_eagerly():
        self.add_update(mean_update, inputs=True)
        self.add_update(variance_update, inputs=True)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    outputs = nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    if not context.executing_eagerly() and original_training_value is None:
      outputs._uses_learning_phase = True  # pylint: disable=protected-access
    return outputs

  # override the _assign_moving_average method since colocate_with creates problems in distributed settings
  def _assign_moving_average(self, variable, value, momentum):
    # print(variable, value, sep='\n')
    with ops.name_scope(None, 'AssignMovingAvg',
                        [variable, value, momentum]) as scope:
      decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
      if decay.dtype != variable.dtype.base_dtype:
        decay = math_ops.cast(decay, variable.dtype.base_dtype)
      update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
      return state_ops.assign_sub(variable, update_delta, name=scope)

  # overwrite _fused_batch_norm so we can have control over synchronizing
  # moving_{mean, variance} variables during training
  def _fused_batch_norm(self, inputs, training):
    """Returns the output of fused batch norm."""
    beta = self.beta if self.center else self._beta_const
    gamma = self.gamma if self.scale else self._gamma_const

    def _cross_replica_non_fused_batch_norm_training():
      # TODO(panos): assert the data format to be NHWC
      # TODO(panos): make a moments function with distributed synchronization

      def _merge_fn(strategy, per_replica_mean, per_replica_square_mean):
        # per_replica_mean: PerDevice
        # global_mean: Mirrored
        global_mean = strategy.reduce(
            tf.VariableAggregation.SUM,
            per_replica_mean,
            per_replica_mean) 
        global_squared_mean = strategy.reduce(
            tf.VariableAggregation.SUM,
            per_replica_square_mean,
            per_replica_square_mean)
        return global_mean, global_squared_mean

      # dispatch as much computation to each replica
      per_replica_mean = tf.reduce_mean(inputs, axis=(0, 1, 2))
      per_replica_square_mean = tf.reduce_mean(tf.square(inputs), axis=(0, 1, 2))

      replica_context = tf.contrib.distribute.get_tower_context()
      global_mean, global_squared_mean = replica_context.merge_call(
          _merge_fn,
          per_replica_mean / replica_context.num_towers,
          per_replica_square_mean / replica_context.num_towers)
      global_variance = global_squared_mean - tf.square(global_mean)

      inputs_normalized = tf.nn.batch_normalization(
          inputs, global_mean, global_variance, beta, gamma, self.epsilon)

      return inputs_normalized, global_mean, global_variance

    def _fused_batch_norm_training():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          epsilon=self.epsilon,
          data_format=self._data_format)

    def _fused_batch_norm_inference():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=self.moving_mean,
          variance=self.moving_variance,
          epsilon=self.epsilon,
          is_training=False,
          data_format=self._data_format)

    output, mean, variance = tf_utils.smart_cond(
        training, _cross_replica_non_fused_batch_norm_training, _fused_batch_norm_inference)
    if not self._bessels_correction_test_only:
      # Remove Bessel's correction to be consistent with non-fused batch norm.
      # Note that the variance computed by fused batch norm is
      # with Bessel's correction.
      sample_size = math_ops.cast(
          array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
      factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
      variance *= factor

    training_value = tf_utils.constant_value(training)
    if training_value is None:
      momentum = tf_utils.smart_cond(training,
                                     lambda: self.momentum,
                                     lambda: 1.0)
    else:
      momentum = ops.convert_to_tensor(self.momentum)
    if training_value or training_value is None:
      mean_update = self._assign_moving_average(self.moving_mean, mean,
                                                momentum)
      variance_update = self._assign_moving_average(self.moving_variance,
                                                    variance, momentum)
      self.add_update(mean_update, inputs=True)
      self.add_update(variance_update, inputs=True)

    return output

def cross_replica_batch_normalization(inputs, *args, **kwargs):
  fused = kwargs.get('fused')
  if fused is None:
    fused = True

  # inputs = ops.convert_to_tensor(inputs)
  rank = inputs.get_shape().ndims

  if kwargs.get('data_format', DATA_FORMAT_NHWC) not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      kwargs.get('scope'),
      'BatchNorm', [inputs],
      reuse=kwargs.get('reuse'),
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Check that we can use the core layer class.
    assert all([
        kwargs.get('batch_weights') is None,
        kwargs.get('updates_collections', ops.GraphKeys.UPDATE_OPS) is ops.GraphKeys.UPDATE_OPS,
        not kwargs.get('zero_debias_moving_mean', False)]), 'This function cannot be used.'

    # Construct and apply the layer
    axis = 1 if kwargs.get('data_format', DATA_FORMAT_NHWC) == DATA_FORMAT_NCHW else -1
    if not kwargs.get('param_initializers', None):
      param_initializers = {}
    beta_initializer = param_initializers.get('beta',
                                              init_ops.zeros_initializer())
    gamma_initializer = param_initializers.get('gamma',
                                               init_ops.ones_initializer())
    moving_mean_initializer = param_initializers.get(
        'moving_mean', init_ops.zeros_initializer())
    moving_variance_initializer = param_initializers.get(
        'moving_variance', init_ops.ones_initializer())
    if not kwargs.get('param_regularizers', None):
      param_regularizers = {}
    beta_regularizer = param_regularizers.get('beta')
    gamma_regularizer = param_regularizers.get('gamma')
    layer = CrossReplicaBatchNormalization(
        axis=axis,
        momentum=kwargs.get('decay', 0.999),
        epsilon=kwargs.get('epsilon', 0.001),
        center=kwargs.get('center', True),
        scale=kwargs.get('scale', False),
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        trainable=kwargs.get('trainable', True),
        renorm=kwargs.get('renorm', False),
        renorm_clipping=kwargs.get('renorm_clipping'),
        renorm_momentum=kwargs.get('renorm_decay', 0.99),
        adjustment=kwargs.get('adjustment'),
        name=sc.name,
        _scope=sc,
        _reuse=kwargs.get('reuse'),
        fused=fused)
    outputs = layer.apply(inputs, training=kwargs.get('is_training', True))

    # Add variables to collections.
    _add_variable_to_collections(layer.moving_mean, kwargs.get('variables_collections'),
                                 'moving_mean')
    _add_variable_to_collections(layer.moving_variance, kwargs.get('variables_collections'),
                                 'moving_variance')
    if layer.beta is not None:
      _add_variable_to_collections(layer.beta, kwargs.get('variables_collections'), 'beta')
    if layer.gamma is not None:
      _add_variable_to_collections(layer.gamma, kwargs.get('variables_collections'),
                                   'gamma')

    if kwargs.get('activation_fn') is not None:
      outputs = kwargs.get('activation_fn')(outputs)
    return utils.collect_named_outputs(kwargs.get('outputs_collections'), sc.name, outputs)

cross_replica_batch_norm = cross_replica_batch_normalization
