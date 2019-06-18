!!! OUTDATED !!!

Demonstrates some good practics for input pipelines on various datasets.

Input pipeline for a Semantic Segmentation model using TF Data API.

directory with images and labels
 ||
 vv
data reading, batching and preparation
(deal with shapes, dtypes, ranges and mappings)
 ||
 vv
raw input images and labels as tf.Tensor    -->  output1: (rawdata, metadata)
 ||
 vv
preprocessing
 ||
 vv
preprocessed images and labels as tf.Tensor -->  output2: (prodata, metadata)
 ||
 vv
QA: this pipeline must gurantee output rate <= 50 ms per output2
    for prodata of shape: (4 x 512 x 1024 x 3, 4 x 512 x 1024)

Directory structure: (still to be decided), for now paths provided at
otherconfig must be enough: e.g. directory paths for images and labels
and recursively scan those directories for examples (such as Cityscapes)

output1: during prediction, when plotting results the original image
  must be also available, or for saving outputs metadata such as original
  file name is needed.

output2: the actual input to Estimator.

input functions: are called by train, evaluate and predict of a
tf.estimator.Estimator instance as input_fn(config, params).
Note: only these names are checked to be passed, thus the only arguments of
input functions must be 'config' and/or 'params'.

problem definition file: a json file containing a single object with at least
the following key-value pairs:
version: version of problem definition (key reserved for later use)
lids2cids: an array of label ids to class ids mappings: each label id in the
  encoded image is mapped to a class id for classification according to this
  array. Ignoring a label is supported by class id -1. Class ids >=0.
  The validity of the mapping is upon the caller for verification. This
  pair is useful for ignoring selected annotated ids or performing category
  classification.
cids2labels: an array of class ids to labels mappings: each class id gets the
  string label of the corresponding index. Void label should be provided first.
cids2colors: an array of class ids to a 3-element array of RGB colors.
Example: parsed json to Python dictionary:
{"version":1.0,
 "comments":"label image is encoded as png with uint8 pixel values denoting
    the label id, e.g. 0:void, 1:static, 2:car, 3:human, 4:bus",
 "lids2cids":[-1,-1,1,0,2],
 "cids2labels":["void", "human", "car", "bus"],
 "cids2colors":[[0,0,0],[45,67,89],[0,0,255],[140,150,160]]}

This guide includes templates and guidelines for creating input pipelines for semantic segmentation.
Input pipelines:
 i) from dataset directory
 ii) from tfrecords
 iii) from file lists