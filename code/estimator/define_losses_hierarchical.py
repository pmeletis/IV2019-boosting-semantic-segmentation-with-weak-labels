"""Losses for Semantic Segmentation model supporting void labels.
"""

import tensorflow as tf
from utils.utils import _replacevoids
from input_pipelines.utils import get_temp_Nb

# Conventions:
# i)  in the batch first comes the per-pixel dataset and then the per-bbox dataset
# ii) there are two modes:
#     i) l1 classifier doesn't collect loss from weakly labeled datasets (l1 sparse CE, l2 dense CE)
#     ii) it does and we use the suffix "_hot" cause multi-hot labels must be used in that case  (l1 dense CE, l2 dense CE)

def define_losses(mode, predictions, labels, config, params): # pylint: disable=unused-argument
  """
  """

  l1_logits = predictions['l1_logits']
  l1_decisions = predictions['l1_decisions']
  l2_vehicle_logits = predictions['l2_vehicle_logits']
  l2_vehicle_probabilities = predictions['l2_vehicle_probabilities']
  l2_human_logits = predictions['l2_human_logits']
  l2_human_probabilities = predictions['l2_human_probabilities']

  ## generate losses
  if mode == tf.estimator.ModeKeys.EVAL:
    tf.logging.info('Losses for evaluation are not yet implemented (set to 0 for now).')
    return {'total': tf.constant(0.),
            'segmentation': tf.constant(0.),
            'regularization': tf.constant(0.)}

  elif mode == tf.estimator.ModeKeys.TRAIN:

    Nb_per_pixel = get_temp_Nb(config, params.Nb_per_pixel)
    Nb_per_bbox = get_temp_Nb(config, params.Nb_per_bbox)
    Nb_per_image = get_temp_Nb(config, params.Nb_per_image)
    per_pixel_dataset = params.per_pixel_dataset_name
    if per_pixel_dataset == 'vistas':
      cid_l1_vehicle = 49
      cid_l1_human = 19
      per_pixel_cids2l1_cids = tf.cast([
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          19, 19, 19, 20, 21, 22, 23, 24, 25, 26,
          27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
          37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
          47, 48, 49, 49, 49, 49, 49, 49, 49, 49,
          49, 49, 49, 50, 51, 52], tf.int32)
      per_bbox_cids2l1_cids = tf.cast([
          49, 49, 49, 49, 49, 49, 19, 19, 19, 19,
          19, 52, 52, 52, 52], tf.int32)
      # 0: bicycle, 1: boat, 2: bus, 3: car, 4: caravan, 5: motorcycle, 6: on rails,
      # 7: other vehicle, 8: trailer, 9: truck, 10: wheeled slow, 11: void
      per_pixel_cids2vehicle_cids = tf.cast([
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
          11, 11,  0,  1,  2,  3,  4,  5,  6,  7,
           8,  9, 10, 11, 11, 11], tf.int32)
      per_bbox_cids2vehicle_cids = tf.cast(
          [0, 2, 3, 5, 6, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11], tf.int32)
      # 0: person, 1: bicyclist, 2: motorcyclist, 3: other rider, 4: void
      per_pixel_cids2human_cids = tf.cast([
          4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 0,
          1, 2, 3, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4], tf.int32)
      per_bbox_cids2human_cids = tf.cast(
          [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4], tf.int32)
    elif per_pixel_dataset == 'cityscapes':
      cid_l1_vehicle = 12
      cid_l1_human = 11
      per_pixel_cids2l1_cids = tf.cast([
           0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          10,11,11,12,12,12,12,12,12,13], tf.int32)
      per_bbox_cids2l1_cids = tf.cast([
          12, 12, 12, 12, 12, 12, 11, 11, 11, 11,
          11, 13, 13, 13, 13], tf.int32)
      per_pixel_cids2vehicle_cids = tf.cast([
          6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
          6, 6, 6, 0, 1, 2, 3, 4, 5, 6], tf.int32)
      per_bbox_cids2vehicle_cids = tf.cast(
          [5, 2, 0, 4, 3, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6], tf.int32)
      per_pixel_cids2human_cids = tf.cast([
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 0, 1, 2, 2, 2, 2, 2, 2, 2], tf.int32)
      per_bbox_cids2human_cids = tf.cast(
          [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2], tf.int32)

    ## labels for l1 classifier
    # vehicle_labels contain per_pixel (dense) labels and open images (sparse) labels
    per_pixel_labels = labels['prolabels_per_pixel']
    per_bbox_labels = labels['prolabels_per_bbox']
    per_image_labels = labels['prolabels_per_image']
    l1_per_pixel_labels = tf.gather(per_pixel_cids2l1_cids, per_pixel_labels)
    # l1_per_pixel_labels_onehot = tf.one_hot(l1_per_pixel_labels, tf.reduce_max(per_pixel_cids2l1_cids)+1)
    # dummy labels, will be masked during loss computation by weights
    l1_per_bbox_labels = tf.constant(
        -1000, dtype=tf.int32, shape=(Nb_per_bbox, *l1_per_pixel_labels.shape[1:]))
    l1_per_image_labels = tf.constant(
        -1000, dtype=tf.int32, shape=(Nb_per_image, *l1_per_pixel_labels.shape[1:]))
    # l1_per_bbox_labels_hot = _segment_sum(per_bbox_labels, per_bbox_cids2l1_cids, tf.reduce_max(per_bbox_cids2l1_cids)+1)
    l1_labels = tf.concat([l1_per_pixel_labels, l1_per_bbox_labels, l1_per_image_labels], 0)
    # l1_labels_hot = tf.concat([l1_per_pixel_labels_onehot, l1_per_bbox_labels_hot], 0)
    #for sone reason l1_labels_hot is not always in [0, 1] so clip it
    # l1_labels_hot = tf.clip_by_value(l1_labels_hot, 0., 1.)
    # print(l1_labels, Nb_per_pixel, Nb_per_bbox)
    l1_labels = tf.stop_gradient(l1_labels) # tf.int32, (H, W), with indices
    # l1_labels_hot = tf.stop_gradient(l1_labels_hot) # tf.float32, (H, W, C) with multi-hot probs

    # tf.abs tf.ones_like(l1_labels_hot)[..., 0]tf.ones_like(l1_labels_hot)[..., 0] - tf.reduce_sum(l1_labels_hot, axis=-1)
    # debug assertion
    # with tf.control_dependencies([
    #     tf.assert_equal(
    #         tf.reduce_sum(l1_labels_hot, axis=-1),
    #         tf.ones_like(l1_labels_hot)[..., 0])]):
    #         tf.cond()
    #   l1_labels_hot = tf.identity(l1_labels_hot)

    ## labels for the vehicle l2 classifier
    l2_vehicle_per_pixel_labels = tf.gather(per_pixel_cids2vehicle_cids, per_pixel_labels)
    l2_vehicle_per_pixel_labels = tf.one_hot(l2_vehicle_per_pixel_labels, tf.reduce_max(per_pixel_cids2vehicle_cids)+1)
    # _segment_sum strategy: e.g. if one pixel belongs to bbox of human and vehicle
    # for the vehicle classifier 1/2 will remain for supervision and 1/2 will go to void
    l2_vehicle_per_bbox_labels = _segment_sum(per_bbox_labels, per_bbox_cids2vehicle_cids, tf.reduce_max(per_bbox_cids2vehicle_cids)+1)
    l2_vehicle_per_image_labels = _segment_sum(per_image_labels, per_bbox_cids2vehicle_cids, tf.reduce_max(per_bbox_cids2vehicle_cids)+1)

    l2_vehicle_labels = tf.concat([l2_vehicle_per_pixel_labels, l2_vehicle_per_bbox_labels, l2_vehicle_per_image_labels], 0)
    l2_vehicle_labels = tf.stop_gradient(l2_vehicle_labels)

    ## labels for the human l2 classifier
    l2_human_per_pixel_labels = tf.gather(per_pixel_cids2human_cids, per_pixel_labels)
    l2_human_per_pixel_labels = tf.one_hot(l2_human_per_pixel_labels, tf.reduce_max(per_pixel_cids2human_cids)+1)
    l2_human_per_bbox_labels = _segment_sum(per_bbox_labels, per_bbox_cids2human_cids, tf.reduce_max(per_bbox_cids2human_cids)+1)
    l2_human_per_image_labels = _segment_sum(per_image_labels, per_bbox_cids2human_cids, tf.reduce_max(per_bbox_cids2human_cids)+1)
    l2_human_labels = tf.concat([l2_human_per_pixel_labels, l2_human_per_bbox_labels, l2_human_per_image_labels], 0)
    l2_human_labels = tf.stop_gradient(l2_human_labels)

    ## l1 loss: for per_pixel, disabled: [and high-confidence loss for open-images]
    with tf.name_scope("l1_cross_entropy_loss",
                       (l1_logits, l1_per_pixel_labels, l1_per_bbox_labels)) as l1_loss_scope:
      l1_raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=l1_per_pixel_labels,
          logits=l1_logits[:Nb_per_pixel, ...],
          name="l1")
    #   l1_raw_loss_hot = tf.nn.softmax_cross_entropy_with_logits(
    #       labels=l1_labels_hot,
    #       logits=l1_logits,
    #       name="l1")
      l1_per_pixel_weights = tf.cast(l1_per_pixel_labels <= tf.reduce_max(per_pixel_cids2l1_cids)-1, tf.float32)
    #   l1_per_pixel_weights_onehot = 1.0 - l1_per_pixel_labels_onehot[..., -1]
      l1_per_bbox_weights = tf.zeros(tf.shape(per_bbox_labels)[:-1])
      # collect weak labels loss if not void and is useful class
    #   l1_per_bbox_weights_hot = tf.logical_and(
    #       tf.reduce_any(tf.greater(per_bbox_labels[..., :-1], 0.001), axis=-1),
    #       tf.logical_or(
    #           tf.greater(tf.reduce_max(l2_vehicle_probabilities[Nb_per_pixel:, ..., :-1], axis=-1), 0.99),
    #           tf.greater(tf.reduce_max(l2_human_probabilities[Nb_per_pixel:, ..., :-1], axis=-1), 0.99)))
    #   l1_per_bbox_weights_hot = tf.cast(l1_per_bbox_weights_hot, tf.float32)
      l1_weights = tf.concat([l1_per_pixel_weights, l1_per_bbox_weights], 0)
    #   l1_weights_hot = tf.concat([l1_per_pixel_weights_onehot, l1_per_bbox_weights_hot], 0)

      # debug summary
      tf.summary.image('l1_weights', l1_weights[..., tf.newaxis], max_outputs=100, family='debug')
    #   tf.summary.image('l1_weights_hot', l1_weights_hot[..., tf.newaxis], max_outputs=100, family='debug')

    ## l2 losses per classifier: for per_pixel and weak labels (open images)
    # all examples from per_pixel must accumulate loss, but only the examples from open images
    # that are found to be vehicle by the parent (l1) classifier: this is implemented with weights
    with tf.name_scope("l2_cross_entropy_losses",
                       (l2_vehicle_logits, l2_vehicle_labels)) as l2_loss_scope:

      # vehicle l2 classifier
      l2_vehicle_raw_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=l2_vehicle_labels,
          logits=l2_vehicle_logits,
          name="vehicle")

      l2_vehicle_per_pixel_weights = 1.0 - l2_vehicle_labels[:Nb_per_pixel, ..., -1]
      # not_void: pixels that are non_void
      # l1 correct: pixels that l1 classifier found as vehicle and the l2 gt
      #   agrees that may belong to vehicle
      # l2_vehicle_labels[Nb_per_pixel:, ..., -1] due to _segment_sum may have all type of values in [0, 1]
      not_void_weights = tf.greater(1.0 - l2_vehicle_labels[Nb_per_pixel:, ..., -1], 0.01)
      with tf.control_dependencies([l1_decisions]):
        l1_correct_weights = tf.logical_and(
            tf.equal(l1_decisions[Nb_per_pixel:, ...], cid_l1_vehicle),
            tf.greater_equal(tf.reduce_max(l2_vehicle_labels[Nb_per_pixel:, ..., :-1], axis=-1), 0.01))
      l2_vehicle_weak_weights = tf.cast(tf.logical_and(not_void_weights, l1_correct_weights), tf.float32)
      l2_vehicle_weights = tf.concat([l2_vehicle_per_pixel_weights, l2_vehicle_weak_weights], 0)

      tf.summary.image('l2_vehicle_weights', l2_vehicle_weights[..., tf.newaxis], max_outputs=100, family='debug')

      # human l2 classifier
      l2_human_raw_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=l2_human_labels,
          logits=l2_human_logits,
          name="human")

      l2_human_per_pixel_weights = 1.0 - l2_human_labels[:Nb_per_pixel, ..., -1]
      # not_void: pixels that are non_void
      # l1 correct: pixels that l1 classifier found as human and the l2 gt
      #   agrees that may belong to human
      not_void_weights = tf.greater(1.0 - l2_human_labels[Nb_per_pixel:, ..., -1], 0.01)
      with tf.control_dependencies([l1_decisions]):
        l1_correct_weights = tf.logical_and(
            tf.equal(l1_decisions[Nb_per_pixel:, ...], cid_l1_human),
            tf.greater_equal(tf.reduce_max(l2_human_labels[Nb_per_pixel:, ..., :-1], axis=-1), 0.01))
      l2_human_weak_weights = tf.cast(tf.logical_and(not_void_weights, l1_correct_weights), tf.float32)
      l2_human_weights = tf.concat([l2_human_per_pixel_weights, l2_human_weak_weights], 0)

      tf.summary.image('l2_human_weights', l2_human_weights[..., tf.newaxis], max_outputs=100, family='debug')

    ## compute losses
    # l1 accumulates from per_pixel and selectively from Open Images
    l1_seg_loss = tf.losses.compute_weighted_loss(
        l1_raw_loss, weights=l1_per_pixel_weights, scope=l1_loss_scope, loss_collection=None)
    # l1_seg_loss_hot = tf.losses.compute_weighted_loss(
    #     l1_raw_loss_hot, weights=l1_weights_hot, scope=l1_loss_scope, loss_collection=None)
    # l2 accumulates from per_pixel and Open Images
    l2_vehicle_seg_loss = tf.losses.compute_weighted_loss(
        l2_vehicle_raw_loss, weights=l2_vehicle_weights, scope=l2_loss_scope, loss_collection=None)
    l2_human_seg_loss = tf.losses.compute_weighted_loss(
        l2_human_raw_loss, weights=l2_human_weights, scope=l2_loss_scope, loss_collection=None)
    l2_seg_loss = l2_vehicle_seg_loss + l2_human_seg_loss

    print('\n\nweak labels loss coeff. changed to 0.1.\n\n')
    seg_loss = l1_seg_loss + 0.1 * l2_seg_loss
    tf.losses.add_loss(seg_loss)
    reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    tot_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    losses = {'total': tot_loss,
              'l1_segmentation': l1_seg_loss,
              'l1_segmentation_hot': tf.zeros_like(l1_seg_loss), #l1_seg_loss_hot,
              'l2_vehicle_segmentation': l2_vehicle_seg_loss,
              'l2_human_segmentation': l2_human_seg_loss,
              'regularization': reg_loss}

  else:
    assert NotImplementedError(f"mode {mode} is invalid or not yet implemented.")

  return losses

def _segment_sum(labels, segment_ids, num_segments):
  labels_transposed = tf.transpose(labels, (3, 0, 1, 2))
  labels_transformed = tf.unsorted_segment_sum(
    labels_transposed, segment_ids, num_segments)
  labels = tf.transpose(labels_transformed, (1, 2, 3, 0))
  return labels
