
import tensorflow as tf
from utils import resize_images_or_labels
import random
import functools

def resize_images_or_labels_test():
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
  for size in ((128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)): # _random_sizes((10, 3000), k): #
    resized_images = resize_images_or_labels(
        images, size, tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True, mode='max')
    resized_labels = resize_images_or_labels(
        labels, size, tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True, mode='min')
    ri = resized_images.eval()
    rl = resized_labels.eval()
    print(images.shape[1:3], '-->', size, '||', ri.shape[1:3])
    print(labels.shape[1:3], '-->', size, '||', rl.shape[1:3])

def main():
  resize_images_or_labels_test()

if __name__ == "__main__":
  main()
