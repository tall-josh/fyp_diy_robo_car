# NOTE:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
import tensorflow as tf
import os

from generator import DataGenerator as gen

# For training (WILL bin steering annos, and WILL normalize throttle)
# Images are normalized
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from generator import preprocess_normalize_images_bin_annos as process_fn
from generator import prepare_batch_images_and_labels_RAND_MIRROR as prep_batch

# For evaluation (will NOT bin steering annos, and will leave throttle 0-1024)
# Images are normalized
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#from generator import preprocess_normalize_images_only as process_fn
#from generator import prepare_batch_images_and_labels_NO_MIRROR as prep_batch

from utils import *
import numpy as np

def load_graph(frozen_pb, prefix):
  with tf.gfile.GFile(frozen_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # When a frozen graph is restored, the tensors
    # are accessed using:
    #   graph.get_tensor_by_name("prefix/name_scope/name:0")
    # for an example, see the main method below.
    tf.import_graph_def(graph_def, name=prefix)

    return graph

if __name__ == "__main__":
  test = load_dataset("../data/evened_test.txt")
  image_dir   = "../data/clr_120_160/images"
  anno_dir    = "../data/clr_120_160/annotations"
  batch_size  = 50
  NUM_BINS    = 15

  test_gen    = DataGenerator(batch_size=batch_size,
                  data_set=test[:100],
                  image_dir=image_dir,
                  anno_dir=anno_dir,
                  preprocess_fn=process_fn,
                  prepare_batch_fn=prep_batch)

  test_gen.reset(shuffle=False)

  graph = load_graph(frozen_path, prefix="vae")
  for op in graph.get_operations():
    print(op.name)

  x = graph.get_tensor_by_name("vae/x:0")
  b = graph.get_tensor_by_name("vae/beta:0")
  t = graph.get_tensor_by_name("vae/training:0")
  z = graph.get_tensor_by_name("vae/sampling/z:0")

  with tf.Session(graph=graph) as sess:
    images = test_gen.get_next_batch()["original_images"]
    emb, _b, _t = sess.run([z, b, t], feed_dict={x: images})
  print(f"b: {_b}, t: {_t}")
  print(np.shape(emb))
  print(emb)
