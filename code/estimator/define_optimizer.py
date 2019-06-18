import tensorflow as tf

def define_optimizer(global_step, params):
  if params.learning_rate_schedule == 'piecewise_constant':
    learning_rate = tf.train.piecewise_constant(global_step,
                                                params.learning_rate_boundaries,
                                                params.learning_rate_values)
  elif params.learning_rate_schedule == 'polynomial_decay':
    learning_rate = tf.train.polynomial_decay(params.learning_rate_initial,
                                              global_step,
                                              params.num_training_steps,
                                              end_learning_rate=params.learning_rate_final,
                                              power=params.learning_rate_power)
  else:
    print('Unknown option for learning rate schedule.')

  if params.optimizer == 'SGDM':
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           params.momentum,
                                           use_nesterov=params.use_nesterov)
  elif params.optimizer == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    assert False, 'Unknown option for optimizer.'

  return optimizer

  # elif params.learning_rate_schedule=='exponential_decay':
  #   assert False, 'Not tested...'
  #   params.Ne_per_decay = 12
  #   params.num_batches_per_decay = int(params.num_batches_per_epoch * params.Ne_per_decay)
  #   params.learning_rate_dacay_steps = int(params.Ne // params.Ne_per_decay) # results learning_rate_dacay_steps + 1 different learning rates
  #   #params.staircase = False
  #  #elif params.learning_rate_schedule=='polynomial_decay':
  #   #params.end_learning_rate = 0.001
  #   #params.Ne_per_decay = params.Ne - 2 # 2 epochs for training with end_learning_rate
  #   #params.num_batches_per_decay = int(params.num_batches_per_epoch * params.Ne_per_decay)
  #   #params.power = 0.6
