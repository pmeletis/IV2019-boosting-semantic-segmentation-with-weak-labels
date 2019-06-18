"""Transforms a Tensorflow binary .pb graph definition to a text .pbtxt graph definition.
"""

import tensorflow as tf
import sys
import os

def main(argv):
  path_in, path_out = argv[0], argv[1]
  with tf.Session() as sess, tf.gfile.FastGFile(path_in, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def)
    tf.train.write_graph(sess.graph, path_out, 'inference_graph.pbtxt', as_text=True)

if __name__ == '__main__':
  main(sys.argv[1:])
