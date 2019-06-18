"""
Keys to features specification for transforming image based datasets
for semantic segmentation to tfrecords.
"""

import tensorflow as tf

KEYS2FEATURES = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'image/format':  tf.FixedLenFeature((), tf.string, default_value=None),
    'image/dtype':   tf.FixedLenFeature((), tf.string, default_value=None),
    'image/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'image/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'label/format':  tf.FixedLenFeature((), tf.string, default_value=None),
    'label/dtype':   tf.FixedLenFeature((), tf.string, default_value=None),
    'label/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'label/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    }
