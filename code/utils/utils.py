
import copy, argparse, collections, json, os, glob, sys
import numpy as np
import tensorflow as tf


class SemanticSegmentationArguments(object):
  """Example class for how to collect arguments for command line execution.
  """

  def __init__(self, mode=None):
    self._parser = argparse.ArgumentParser()
    self.add_system_arguments()
    self.add_tf_arguments()
    if mode == tf.estimator.ModeKeys.PREDICT:
      self.add_inference_arguments()
    elif mode == tf.estimator.ModeKeys.TRAIN:
      self.add_train_arguments()
    elif mode == tf.estimator.ModeKeys.EVAL:
      self.add_evaluate_arguments()
    else:
      pass

  @property
  def argparser(self):
    return self._parser

  def parse_args(self, argv):
    # parse all arguments and add manually additional arguments
    self.args = self._parser.parse_args(argv)

    return self.args

  def add_system_arguments(self):
    # hs -> ||                                 SYSTEM                                       || -> hs/s
    # hs -> || hn -> ||                   LEARNABLE NETWORK                     || -> hn/sn || -> hs/s
    # hs -> || hn -> ||  hf -> FEATURE EXTRACTOR -> hf/sfe -> [UPSAMPLER -> hf] || -> hn/sn || -> hs/s
    # input || image || [tile] -> batch ->               supervision -> [stich] ||  labels  || output
    # self._parser.add_argument('--stride_system', type=int, default=1, help='Output stride of the system. Use 1 for same input and output dimensions.')
    # self._parser.add_argument('--stride_network', type=int, default=1, help='Output stride of the network. Use in case labels have different dimensions than output of learnable network.')
    self._parser.add_argument('--height_system', type=int, default=None,
                              help='Height of input images to the system. If None arbitrary height is supported (for now effective only during inference, training requires predefined spatial size).')
    self._parser.add_argument('--width_system', type=int, default=None,
                              help='Width of input images to the system. If None arbitrary width is supported (for now effective only during inference, training requires predefined spatial size).')
    # self._parser.add_argument('--height_network', type=int, default=512, help='Height of input images to the trainable network.')
    # self._parser.add_argument('--width_network', type=int, default=1024, help='Width of input images to the trainable network.')
    self._parser.add_argument('--height_feature_extractor', type=int, default=512, help='Height of feature extractor images. If height_feature_extractor != height_network then it must be its divisor for patch-wise training.')
    self._parser.add_argument('--width_feature_extractor', type=int, default=1024, help='Width of feature extractor images. If width_feature_extractor != width_network then it must be its divisor for patch-wise training.')

  def add_tf_arguments(self):
    # general flags
    #self._parser.add_argument('--log_every_n_steps', type=int, default=100, help='The frequency, in terms of global steps, that the loss and global step and logged.')
    #self._parser.add_argument('--summarize_grads', default=False, action='store_true', help='Whether to mummarize gradients.')
    self._parser.add_argument('--enable_xla', action='store_true', help='Whether to enable XLA accelaration.')

  def add_train_arguments(self):
    """Arguments for training.

    TFRecords requirements...
    """
    # general configuration flags (in future will be saved in otherconfig)
    self._parser.add_argument('log_dir', type=str,
                              help='Directory for saving checkpoints, settings, graphs and training statistics.')
    self._parser.add_argument('per_pixel_dataset_name', type=str, choices=['cityscapes', 'vistas'])
    self._parser.add_argument('--Ntrain', type=int, default=2975,
                              help='Temporary parameter for the number of training examples in the tfrecords for computing the number of steps per epoch.')
    self._parser.add_argument('--init_ckpt_path', type=str, default='/media/panos/data/pretrained/resnet_v1_50_official.ckpt',
                              help='Provide an empty string for training from scratch. If a checkpoint is provided and log_dir is empty, same variables between checkpoint and the model will be initiallized from this checkpoint. Otherwise, training will continue from the latest checkpoint in log_dir according to tf.Estimator. If you want to initialize partially from this checkpoint delete of modify names of variables in the checkpoint.')
    self._parser.add_argument('--training_problem_def_path', type=str,
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('--save_checkpoints_steps', type=int, default=None,
                              help='Save checkpoint every save_checkpoints_steps steps. If None uses number of examples, epochs and batch size to save one checkpoint per epoch.')
    self._parser.add_argument('--save_summaries_steps', type=int, default=120,
                              help='Save summaries every save_summaries_steps steps.')

    self._parser.add_argument('--train_void_class', action='store_true',
                              help='If provided an extra class with all unlabeled pixels (-1 in label ids) will be trained.')

    # self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
    #                           help='[NOT IMPLEMENTED] Problem definition json file for inference. If provided it will be used instead of training_problem_def for inference during training.')
    # self._parser.add_argument('--evaluation_problem_def_path', type=str, default=None,
    #                           help='[NOT IMPLEMENTED] Problem definition json file for evaluation during training. If provided it will be used instead of training_problem_def for evaluation during training.')
    # self._parser.add_argument('--collage_image_summaries', action='store_true', help='Whether to collage input and result images in summaries. Inputs and results won\'t be collaged if they have different sizes.')

    # optimization and losses flags (in future will be saved in hparams)
    self._parser.add_argument('--Ne', type=int, default=17, help='Number of epochs to train for.')
    self._parser.add_argument('--Nb', type=int, default=4, help='Number of examples per batch. Increase with number of classes.')
    self._parser.add_argument('--learning_rate_schedule', type=str, default='piecewise_constant',
                              choices=['piecewise_constant', 'polynomial_decay'],
                              help='Learning rate schedule.')
    self._parser.add_argument('--learning_rate_initial', type=float, default=0.01, help='Initial learning rate.')
    # used in piecewise_constant
    # Tried settings for ResNet-50 with SGDM:
    # @512x512: 1) Nb = 4 in 1 GPU: [8, 15, 17], [0.01, 0.005, 0.0025]
    #              <=8 (if >8 starts to diverge (for bn_decay=0.9, earlier for 0.99, kok))
    #           2) Nb = 12 in 3 GPUs (synchronous): [10, 17, 19], [0.1, 0.05, 0.025]
    self._parser.add_argument('--learning_rate_boundaries', type=int, default=[8, 15, 17], nargs='*',
                              help='Boundaries in epochs. If not provided, the final boundary is computed '
                                   'by Ne - learning_rate_boundaries[-1]. User has to set --Ne accordingly.')
    lr_properties_group = self._parser.add_mutually_exclusive_group()
    lr_properties_group.add_argument('--learning_rate_decay', type=float, # default=0.5,
                                     help='Decay rate for learning rate. Has priority over learning_rate_values '
                                          'if both are provided.')
    lr_properties_group.add_argument('--learning_rate_values', type=float, nargs='*', # default=[0.01, 0.005, 0.0025]
                                     help='Values for each plateau. Is ignored if learning_rate_decay is provided.')
    # used in polynomial_decay
    self._parser.add_argument('--learning_rate_decay_steps', type=float, default=0.5, help='Refer to TF docs.')
    self._parser.add_argument('--learning_rate_final', type=float, default=0.5, help='Refer to TF docs.')
    self._parser.add_argument('--learning_rate_power', type=float, default=0.9, help='Refer to TF docs.')
    # optimizer
    self._parser.add_argument('--optimizer', type=str, default='SGDM', choices=['SGD', 'SGDM'], help='Stochastic Gradient Descent optimizer with or without Momentum.')
    self._parser.add_argument('--ema_decay', type=float, default=0.9, help='If >0 additionally save exponential moving averages of training variables with this decay rate.')
    self._parser.add_argument('--regularization_weight', type=float, default=0.00017, help='Weight for the L2 regularization losses in the total loss (decrease when using smaller batch (Nb or image dims)).')
    self._parser.add_argument('--bootstrapping_percentage', type=int, default=-1,
                              help='Percentage of pixels to bootstrap while computing the loss. -1 disables bootstrapping. Set it closer to 100 the more unlabeled pixels a dataset has.')
    # only for Momentum optimizer
    self._parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGDM.')
    self._parser.add_argument('--use_nesterov', action='store_true', help='Enable Nesterov acceleration for SGDM.')
    self._parser.add_argument('--distribute', action='store_true', help='Distribute training across available GPUs on the same machine according to MirrorStrategy. Take care of the schedules (like learning rate\'s), since the effective batch size will be num_gpus * Nb.')

  def add_inference_arguments(self):

    # saved model arguments: log_dir, [ckpt_path], training_problem_def_path
    self._parser.add_argument('log_dir', type=str, default=None,
                              help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be loaded from this directory by default, unless ckpt_path is provided.')
    self._parser.add_argument('--ckpt_path', type=str, default=None,
                              help='If provided, this checkpoint (if exists) will be used.')
    self._parser.add_argument('training_problem_def_path', type=str,
                              help='Problem definition json file that the model is trained with. For required fields refer to help.')

    # inference arguments: prediction_dir, [results_dir], [inference_problem_def_path],
    #                      [plotting], [export_color_decisions], [export_lids_images]
    self._parser.add_argument('predict_dir', type=str, default=None,
                              help='Directory to scan for media recursively and to write results under created results directory with the same directory structure. For supported media files check help.')
    self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
                              help='Problem definition json file for inference. If provided it will be used instead of training_problem_def. For required fields refer to help.')
    self._parser.add_argument('--replace_voids', action='store_true',
                              help='Whether to replace void labeled pixels with the second most probable class (effective only when void (-1) is provided in lids2cids field in training problem definition). Enable only for official prediction/evaluation as it uses a time consuming function.')

    # SemanticSegmentation and system arguments:  [Nb], [restore_emas]
    self._parser.add_argument('--Nb', type=int, default=1,
                              help='Number of examples per batch.')
    self._parser.add_argument('--restore_emas', action='store_true',
                              help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')
    self._parser.add_argument('--train_void_class', action='store_true',
                              help='Provide this flag if training was also done with this flag.')

    # consider for adding in the future arguments
    # self._parser.add_argument('--export_probs', action='store_true', help='Whether to export probabilities results.')
    # self._parser.add_argument('--export_for_algorithm_evaluation', action='store_true', help='Whether to plot and export using the algorithm input size (h,w).')

  def add_evaluate_arguments(self):
    self._parser.add_argument('log_dir', type=str, default=None,
                              help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be evaluated from this directory by default, unless ckpt_path or evall_all_ckpts are provided.')
    self._parser.add_argument('--eval_all_ckpts', action='store_true',
                              help='Whether to evaluate all checkpoints in log_dir. It has priority over --ckpt_path argument.')
    self._parser.add_argument('--ckpt_path', type=str, default=None,
                              help='If provided, this checkpoint (if exists) will be evaluated.')
    self._parser.add_argument('Neval', type=int,
                              help='Temporary parameter for the number of evaluated examples.')
    self._parser.add_argument('training_problem_def_path', type=str,
                              help='Problem definition json file that the model is trained with. For required fields refer to help.')
    self._parser.add_argument('--evaluation_problem_def_path', type=str, default=None,
                              help='Problem definition json file for evaluation. If provided it will be used instead of training_problem_def. For required fields refer to help.')
    self._parser.add_argument('--replace_voids', action='store_true',
                              help='Replace void pixel decisions with the next most probable (not ignored) label. During training, if unlabeled pixels exist in ground truth (corresponding to label "-1" in lids2cids field in training problem definition) predictions may include a void label. During evaluation some labels are ignored (corresponding to label "-1" in training_cids2evaluation_cids field in evaluation problem definition), thus predictions may have void decisions.')
    self._parser.add_argument('--train_void_class', action='store_true',
                              help='Provide this flag if training was also done with this flag.')
    
    # self._parser.add_argument('eval_steps', type=int, help='500 for cityscapes val, 2975 for cityscapes train.')
    self._parser.add_argument('--Nb', type=int, default=1,
                              help='Number of examples per batch.')
    self._parser.add_argument('--restore_emas', action='store_true',
                              help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')

    # consider for future
    # self._parser.add_argument('--results_dir', type=str, default=None,
    #                           help='If provided evaluation results will be written to this directory, otherwise they will be written under a created results directory under log_dir.')

  def add_export_frozen_graph_arguments(self):

    # saved model arguments: log_dir, [ckpt_path], training_problem_def_path
    self._parser.add_argument('log_dir', type=str, default=None,
                              help='Logging directory where the exported graph will be saved.')
    self._parser.add_argument('ckpt_path', type=str, default=None,
                              help='The checkpoint filepath from which variables will be restored.')
    self._parser.add_argument('training_problem_def_path', type=str,
                              help='Problem definition json file that the model is trained with. For required fields refer to help.')

    # inference arguments: prediction_dir, [results_dir], [inference_problem_def_path],
    #                      [plotting], [export_color_decisions], [export_lids_images]
    # self._parser.add_argument('predict_dir', type=str, default=None,
    #                           help='Directory to scan for media recursively and to write results under created results directory with the same directory structure. For supported media files check help.')
    # self._parser.add_argument('--results_dir', type=str, default=None,
    #                           help='If provided results will be written to this directory.')
    # self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
    #                           help='Problem definition json file for inference. If provided it will be used instead of training_problem_def. For required fields refer to help.')
    # self._parser.add_argument('--plotting', action='store_true',
    #                           help='Whether to plot results.')
    # self._parser.add_argument('--timeout', type=float, default=10.0,
    #                           help='Timeout for continuous plotting, if plotting flag is provided.')
    # self._parser.add_argument('--export_color_decisions', action='store_true',
    #                           help='Whether to export color image results.')
    # self._parser.add_argument('--export_lids_images', action='store_true',
    #                           help='Whether to export label ids image results. Label ids are defined in {training,inference}_problem_def_path.')
    # self._parser.add_argument('--replace_void_decisions', action='store_true',
    #                           help='[NOT IMPLEMENTED] Whether to replace void labeled pixels with the second most probable class (effective only when void (-1) is provided in lids2cids field in training problem definition). Enable only for official prediction/evaluation as it uses a time consuming function.')
    self._parser.add_argument('--preserve_aspect_ratio', action='store_true',
                              help='Whether to resize respecting aspect ratio.')

    # SemanticSegmentation and system arguments:  [Nb], [restore_emas]
    # self._parser.add_argument('--Nb', type=int, default=1,
    #                           help='Number of examples per batch.')
    self._parser.add_argument('--restore_emas', action='store_true',
                              help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')

    # consider for adding in the future arguments
    # self._parser.add_argument('--export_probs', action='store_true', help='Whether to export probabilities results.')
    # self._parser.add_argument('--export_for_algorithm_evaluation', action='store_true', help='Whether to plot and export using the algorithm input size (h,w).')

  def add_predict_image_arguments(self):

    # saved model arguments: log_dir, [ckpt_path], training_problem_def_path
    self._parser.add_argument('log_dir', type=str, default=None,
                              help='Logging directory where the exported graph will be saved.')
    self._parser.add_argument('training_problem_def_path', type=str,
                              help='Problem definition json file that the model is trained with. For required fields refer to help.')
    self._parser.add_argument('ckpt_path', type=str, default=None,
                              help='The checkpoint filepath from which variables will be restored.')
    # inference arguments: prediction_dir, [results_dir], [inference_problem_def_path],
    #                      [plotting], [export_color_decisions], [export_lids_images]
    self._parser.add_argument('predict_dir', type=str, default=None,
                              help='Directory to scan for media recursively and to write results under created results directory with the same directory structure. For supported media files check help.')
    self._parser.add_argument('--results_dir', type=str, default=None,
                              help='If provided results will be written to this directory.')
    self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
                              help='Problem definition json file for inference. If provided it will be used instead of training_problem_def. For required fields refer to help.')
    self._parser.add_argument('--plotting', action='store_true',
                              help='Whether to plot results.')
    self._parser.add_argument('--timeout', type=float, default=10.0,
                              help='Timeout for continuous plotting, if plotting flag is provided.')
    self._parser.add_argument('--export_color_decisions', action='store_true',
                              help='Whether to export color image results.')
    self._parser.add_argument('--export_lids_images', action='store_true',
                              help='Whether to export label ids image results. Label ids are defined in {training,inference}_problem_def_path.')
    self._parser.add_argument('--replace_voids', action='store_true',
                              help='Whether to replace void labeled pixels with the second most probable class (effective only when void (-1) is provided in lids2cids field in training problem definition). Enable only for official prediction/evaluation as it uses a time consuming function.')

    # SemanticSegmentation and system arguments:  [Nb], [restore_emas]
    # self._parser.add_argument('--Nb', type=int, default=1,
    #                           help='Number of examples per batch.')
    self._parser.add_argument('--restore_emas', action='store_true',
                              help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')

    # consider for adding in the future arguments
    # self._parser.add_argument('--export_probs', action='store_true', help='Whether to export probabilities results.')
    # self._parser.add_argument('--export_for_algorithm_evaluation', action='store_true', help='Whether to plot and export using the algorithm input size (h,w).')

def print_tensor_info(tensor):
  print(f"debug:{tensor.op.name}: {tensor.get_shape().as_list()} {tensor.dtype}")

def get_unique_variable_by_name_without_creating(name):
  variables = [v for v in tf.global_variables() + tf.local_variables() if name==v.op.name]
  assert len(variables)==1, f"Found {len(variables)} variables for name {name}."
  return variables[0]

def get_unique_tensor_by_name_without_creating(name):
  tensors = [t for t in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()) if name==t.name]
  assert len(tensors)==1, f"Found {len(tensors)} tensors."
  return tensors[0]

def get_saveable_objects_list(graph):
  return (graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + graph.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))

def ids2image(ids, palette):
  # ids: Nb x H x W, with elements in [0,K-1]
  # palette: K x 3, tf.uint8
  # returns: Nb x H x W x 3, tf.uint8
  
  # TODO: add type checking
  return tf.gather_nd(palette, tf.expand_dims(ids, axis=-1))

def almost_equal(num1, num2, error=10**-3):
  return abs(num1-num2) <= error

def _replacevoids(mappings):
  # replace voids with max id + 1
  max_m = max(mappings)
  return [m if m!=-1 else max_m+1 for m in mappings]

# class MediaLoader(object):
#   image_formats = ['png','PNG','jpg','JPG','ppm','PPM']
#   video_formats = ['mp4','MP4','avi','AVI']
  
#   class Media(object):
#     def __init__(self, fpath):
#       self.fpath = fpath
#       _, self.fname, self.fformat = MediaLoader.split_path(self.fpath)
#       if any(self.fformat in imf for imf in MediaLoader.image_formats):
#         self.ftype = 'image'
#       elif any(self.fformat in vif for vif in MediaLoader.video_formats):
#         self.ftype = 'video'
#       else:
#         assert False, 'Error in code of MediaLoader...'
#       self.media = None
#       self.read_frames = []
    
#     def read(self):
#       if self.ftype=='image':
#         self.media = misc.imread(self.fpath)
#         return [self.media]
#       elif self.ftype=='video':
#         self.media = mp.VideoFileClip(self.fpath)
#         return self.media.iter_frames()
  
#   @staticmethod
#   def get_known_filepaths(folder_path):
#     # scan folder for known image and video files
#     file_paths = []
#     for suffix in MediaLoader.image_formats + MediaLoader.video_formats:
#       file_paths += glob.glob(folder_path + '/*.' + suffix)
#     return file_paths
  
#   @staticmethod
#   def split_path(path):
#     # filepath = head/root.ext[1:]
#     head, tail = os.path.split(path)
#     root, ext = os.path.splitext(tail)
#     return head, root, ext[1:]
      
#   def __init__(self, folder_path):
#     self.folder_path = folder_path
#     self.fpaths = MediaLoader.get_known_filepaths(folder_path)
#     print('debug:N', len(self.fpaths))
#     self.medias = []
    
#   def _next_media(self):
#     # generator which yields iterator for each filename
#     for fp in self.fpaths:
#       media = MediaLoader.Media(fp)
#       self.medias.append(media)
#       yield self.medias[-1]
  
#   def next_frame(self):
#     for media in self._next_media():
#       #print('debug:medias:', len(self.medias))
#       for frame in media.read():
#         media.read_frames.append(frame)
#         #print('debug:frames:', len(media.read_frames))
#         yield media.read_frames[-1]

# class CurrentMedia(object):
#   def __init__(self):
#     self.media = None
#     self.ftype = 'image' # trick to pop media the first time
#     self.frame = None
  
#   def update(self, mloader):
#     if self.ftype=='image' or (self.ftype=='video' and len(self.media.read_frames)==0):
#       self.media = mloader.medias.pop(2)
#       self.ftype = self.media.ftype
#     self.frame = self.media.read_frames.pop(0)

# safe_div from tensorflow/python/ops/losses/losses_impl.py
def safe_div(num, den, name="safe_div"):
  """Computes a safe divide which returns 0 if the den is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    num: An arbitrary `Tensor`.
    den: `Tensor` whose shape matches `num` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the num divided by the den.
  """
  return tf.where(tf.greater(den, 0),
                  tf.div(num,
                         tf.where(tf.equal(den, 0),
                                  tf.ones_like(den), den)),
                  tf.zeros_like(num),
                  name=name)

def print_metrics_from_confusion_matrix(
    cm,
    labels=None,
    printfile=None,
    printcmd=False,
    summary=False):
  # cm: numpy, 2D, square, np.int32 array, not containing NaNs
  # labels: python list of labels
  # printfile: file handler or None to print metrics to file
  # printcmd: True to print a summary of metrics to terminal
  # summary: if printfile is not None, prints only a summary of metrics to file
  
  # sanity checks
  assert isinstance(cm, np.ndarray), 'Confusion matrix must be numpy array.'
  cms = cm.shape
  assert all([cm.dtype==np.int32,
              cm.ndim==2,
              cms[0]==cms[1],
              not np.any(np.isnan(cm))]), (
                f"Check print_metrics_from_confusion_matrix input requirements. "
                f"Input has {cm.ndim} dims, is {cm.dtype}, has shape {cms[0]}x{cms[1]} "
                f"and may contain NaNs.")
  if not labels:
    labels = ['unknown']*cms[0]
  assert len(labels)==cms[0], (
    f"labels ({len(labels)}) must be enough for indexing confusion matrix ({cms[0]}x{cms[1]}).")
  #assert os.path.isfile(printfile), 'printfile is not a file.'
  
  # metric computations
  global_accuracy = np.trace(cm)/np.sum(cm)*100
  # np.sum(cm,1) can be 0 so some accuracies can be nan
  accuracies = np.diagonal(cm)/np.sum(cm,1)*100
  # denominator can be zero only if #TP=0 which gives nan, trick to avoid it
  inter = np.diagonal(cm)
  union = np.sum(cm,0)+np.sum(cm,1)-np.diagonal(cm)
  ious = inter/np.where(union>0,union,np.ones_like(union))*100
  notnan_mask = np.logical_not(np.isnan(accuracies))
  mean_accuracy = np.mean(accuracies[notnan_mask])
  mean_iou = np.mean(ious[notnan_mask])
  
  # reporting
  log_string = "\n"
  log_string += f"Global accuracy: {global_accuracy:5.2f}\n"
  log_string += "Per class accuracies (nans due to 0 #Trues) and ious (nans due to 0 #TPs):\n"
  for k,v in {l:(a,i,m) for l,a,i,m in zip(labels, accuracies, ious, notnan_mask)}.items():
    log_string += f"{k:<30s}  {v[0]:>5.2f}  {v[1]:>5.2f}  {'' if v[2] else '(ignored in averages)'}\n"
  log_string += f"Mean accuracy (ignoring nans): {mean_accuracy:5.2f}\n"
  log_string += f"Mean iou (ignoring accuracies' nans but including ious' 0s): {mean_iou:5.2f}\n"

  if printcmd:
    print(log_string)

  if printfile:
    if summary:
      printfile.write(log_string)
    else:
      print(f"{global_accuracy:>5.2f}",
            f"{mean_accuracy:>5.2f}",
            f"{mean_iou:>5.2f}",
            accuracies,
            ious,
            file=printfile)

def split_path(path):
  # filepath = <head>/<tail>
  # filepath = <head>/<root>.<ext[1:]>
  head, tail = os.path.split(path)
  root, ext = os.path.splitext(tail)
  return head, root, ext[1:]

def count_non_i(int_lst, i):
  # counts the number of integers not equal to i in the integer list int_lst
  
  # assertions
  assert isinstance(int_lst, list), 'int_lst is not a list.'
  assert all([isinstance(e, int) for e in int_lst]), 'Not integer int_list.'
  assert isinstance(i, int), 'Not integer i.'

  # implementation
  return len(list(filter(lambda k: k != i, int_lst)))

# def map_metrics_to_problem(metrics, problem1toproblem2):
#   # if a net should be evaluated with problem that is not the problem with
#   #   which it was trained for, then the mappings from that problem should
#   #   be provided.
#   # metrics is a dictionary, for now only confusion matrix metric is known
#   #   how to be mapped
#   # problem1TOproblem2: list of ints which map ids of problem1 to 
#   #   ids of problem2

#   assert set(metrics.keys()) == set(['confusion_matrix', 'loss', 'global_step']), (
#     f"Only confusion matrix metric is known how to be converted.")
  
#   # confusion matrix conversion
#   # TODO: confusion matrix type and shape assertions
#   # np.set_printoptions(threshold=np.nan)
#   old_cm = metrics['confusion_matrix']
#   # print(old_cm)
#   assert old_cm.shape[0] == len(problem1toproblem2), (
#     f"Mapping lengths should me equal.")
  
#   temp_shape = (max(problem1toproblem2)+1,old_cm.shape[1])
#   temp_cm = np.zeros(temp_shape, dtype=np.int64)
  
#   # mas noiazei to kathe kainourio apo poio palio pairnei:
#   #   i row of the new cm takes from rows of the old cm with indices:from_indices
#   for i in range(temp_shape[0]):
#     from_indices = [k for k, x in enumerate(problem1toproblem2) if x == i]
#     # print(from_indices)
#     for fi in from_indices:
#       temp_cm[i,:] += old_cm[fi,:].astype(np.int64)
  
#   # oi grammes athroistikan kai tora tha athroistoun kai oi stiles
#   new_shape = (max(problem1toproblem2)+1,max(problem1toproblem2)+1)
#   new_cm = np.zeros(new_shape, dtype=np.int64)
#   for j in range(new_shape[1]):
#     from_indices = [k for k, x in enumerate(problem1toproblem2) if x == j]
#     # print(from_indices)
#     for fi in from_indices:
#       new_cm[:,j] += temp_cm[:,fi]
  
#   # print(new_cm)
#   metrics['confusion_matrix'] = new_cm
  
#   return metrics

def tensor_shape(tensor, rank):
  """Returns the dimensions of tensor.
  Adapted from (fdcb7ca) tensorflow/tensorflow/python/ops/image_ops_impl.py/_ImageDimensions

  Args:
    tensor: A rank-D Tensor.
    rank: The expected rank of the tensor

  Returns:
    A tuple of the dimensions of the input tensor. Dimensions that are statically
    known are python integers, otherwise they are integer scalar tensors.
  """
  if tensor.get_shape().is_fully_defined():
    return tuple(tensor.get_shape().as_list())
  else:
    static_shape = tensor.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor), rank)
    return tuple(
        s if s is not None else d for s, d in zip(static_shape, dynamic_shape))

def resize_images_or_labels(features,
                            candidate_size,
                            method,
                            preserve_aspect_ratio=False,
                            mode=None,
                            crop=None):
  """Resizes `features` spatial dimensions to target `candidate_size` using method `method`.
  If `preserve_aspect_ratio` is `True` the output's spatial dimensions may not be equal
  to `candidate_size` and the algorithm will preserve aspect ratio according to `mode`.
  `mode` should be either 'max' (default): `features` will be resized to the tightest
  spatial dimensions so that 'candidate_size' fits in them, or 'min': `features` will be resized
  to the tightest spatial dimensions so it fits in 'candidate_size'. In case of preserving the
  aspect ratio `crop` argument can be "random" so that the resized features have exactly the
  `candidate_shape`.

  This function is intended for resizing images and labels for semantic segmentation.
  `features` shape is expected to be: Nb x H x W [x C]

  Args:
    features: tf.float32 with shape (?, ?, ?, ?), or tf.int32 with shape (?, ?, ?)
    candidate_size: Python int tuple (target_height, target_width)
    method: one of tf.image.ResizeMethod
    preserve_aspect_ratio: if set the output spatial dimensions may not be 'candidate_size'
    mode: 'max' (default) or 'min'
    crop: one {`None`, "random"}, if not `None` the output size is guaranteed to be the
      candidate_size by spatial cropping
  
  Return:
    resized `features` using tf.image.resize_images
  """

  assert features.shape.with_rank_at_least(3)
  is_image = features.dtype == tf.float32 and features.shape.ndims == 4
  is_label = features.dtype == tf.int32 and features.shape.ndims == 3
  assert is_image or is_label, f"'features' ({features}) don't comply to specifications."

  target_size = candidate_size

  if preserve_aspect_ratio:
    if mode is None:
      mode = 'max'
    assert mode in ['max', 'min'], f"'mode' {mode} is not supported."

    features_shape = tensor_shape(features, features.shape.ndims)
    features_height, features_width = features_shape[1:3]
    target_height, target_width = candidate_size

    # implicit casting to tf.float64
    scale_factor_height = target_height / features_height
    scale_factor_width = target_width / features_width

    if mode == 'max':
      scale_factor = tf.maximum(scale_factor_height, scale_factor_width)
    elif mode == 'min':
      scale_factor = tf.minimum(scale_factor_height, scale_factor_width)

    def _target_dim(features_dim):
      return tf.cast(tf.ceil(scale_factor * tf.cast(features_dim, tf.float64)), tf.int32)
    target_size = tuple(map(_target_dim, (features_height, features_width)))

  if is_image:
    resized_features = tf.image.resize_images(features, target_size, method)
  elif is_label:
    resized_features = tf.image.resize_images(
        features[..., tf.newaxis], target_size, method)[..., 0]
  
  if preserve_aspect_ratio:
    if crop == "random":
      crop_shape = (features_shape[0], *candidate_size) + ((features_shape[3],) if is_image else ())
      resized_features = tf.random_crop(resized_features, crop_shape)
    elif crop:
      # more cropping methods to be added soon
      pass
  
  # TODO(panos): statically define shape if cropping is enabled

  return resized_features
