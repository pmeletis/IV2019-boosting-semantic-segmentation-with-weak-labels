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
