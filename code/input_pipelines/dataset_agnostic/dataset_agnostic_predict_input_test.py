import tensorflow as tf
from dataset_agnostic_predict_input import _resize_respecting_aspect_ratio
import random
import functools

tf.InteractiveSession()

images = tf.ones((4, 512, 1024, 3), dtype=tf.float32)
labels = tf.zeros((4, 512, 1024), dtype=tf.int32)

# number of random sizes
k = 20

def _random_sizes(rng, k):
  # returns `k` random sizes from range `range`
  # range: tuple of ints
  randint_in_range = functools.partial(random.randint, rng[0], rng[1])
  for _ in range(k):
    random_size = (randint_in_range(), randint_in_range())
    yield random_size

print('input size', '-->', 'feature extractor size', '||', 'smart size')
for size in ((1024, 2048),): # _random_sizes((10, 3000), k):
  resized_images = _resize_respecting_aspect_ratio(images, size, tf.image.ResizeMethod.BILINEAR)
  resized_labels = _resize_respecting_aspect_ratio(labels, size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  ri = resized_images.eval()
  rl = resized_labels.eval()
  print(images.shape[1:3], '-->', size, '||', ri.shape[1:3])
