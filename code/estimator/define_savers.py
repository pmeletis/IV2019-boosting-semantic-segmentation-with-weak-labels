import tensorflow as tf

def train_saver(config, params, scope='saver'):
  # saver intended for saving vars only during training (not restoring)
  # WARNING: for now TF does not separate between saver and loaders: that is it
  # uses savers also to load a model, thus variables in exclude must be initialized manually

  # next two lines were added for distributed debugging
  if params.distribute:
    tower_context = tf.contrib.distribute.get_tower_context()
    assert tower_context
    print(f"Tower {tower_context.tower_id}: train_saver returns None for distributed training.")
    return None

  if params.init_ckpt_path:
    with tf.name_scope(scope):
      # exclude = ['train_ops']
      exclude = []
      var_list = []
      for var in tf.global_variables():
        for exc in exclude:
          if exc.op.name in var.op.name:
            break
        else:
          var_list.append(var)

      # print('gv', len(tf.global_variables()), 'sv', len(var_list))

      saver = tf.train.Saver(var_list,
                             sharded=True,
                             max_to_keep=config.keep_checkpoint_max,
                             save_relative_paths=True)
  else:
    saver = None

  return saver

def predict_saver(config, params, scope='saver'):
  del config
  # saver intended for restoring vars during evaluation or inference
  with tf.name_scope(scope):
    # mappings from checkpoint variable name to graph variable
    var_dict = dict()
    for mv in tf.model_variables():
      k = mv.op.name
      if params.restore_emas and ('BatchNorm/moving' not in k):
        k = 'exponential_moving_averages/' + k + '/ExponentialMovingAverage'
      # temp solution for recovering names before renaming l2_features to l2_vehicle_features
      # tf.logging.warn('\nTemp solution for recovering names from checkpoints in predict saver.\n')
      # if 'adaptation_module' in k:
      #   k = k.replace('l2_vehicle_features', 'l2_features')
      # if 'softmax_classifier' in k:
      #   k = k.replace('l2_vehicle_logits', 'l2_logits')

      var_dict[k] = mv

    # for now only global_step is in rest_vars
    rest_vars = set(tf.global_variables()).difference(set(var_dict.values()))
    for rv in rest_vars:
      var_dict[rv.op.name] = rv

    saver = tf.train.Saver(var_list=var_dict,
                           sharded=True,
                           save_relative_paths=True)

  return saver

evaluate_saver = predict_saver
export_frozen_graph_saver = predict_saver
