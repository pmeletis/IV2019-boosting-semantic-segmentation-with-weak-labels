import tensorflow as tf

EPSILON = 1E-9

def mean_iou(labels, decisions, num_classes, params):
  # it has been observed that the mean IoU of a training batch is a good
  # estimator of the evaluation final mean IoU, if Nb is at least 4
  flatten = lambda tensor: tf.reshape(tensor, [-1])

  conf_matrix = tf.confusion_matrix(labels=flatten(labels),
                                    predictions=flatten(decisions),
                                    num_classes=num_classes)
  inter = tf.diag_part(conf_matrix)
  union = tf.reduce_sum(conf_matrix, 0) + tf.reduce_sum(conf_matrix, 1) - inter

  inter = tf.cast(inter, tf.float32)
  union = tf.cast(union, tf.float32) + EPSILON
  m_iou = tf.reduce_mean(tf.div(inter, union))

  return m_iou
