# NOTE:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
import tensorflow as tf
import os
import sys
sys.path.append("..")
from vae_data_generator import DataGenerator
from utils import *
import numpy as np
load_dir   = "z_freeze"
frozen_path = os.path.join(load_dir, "FROZEN.pb")
graph = None

def load_graph(frozen_pb):
  with tf.gfile.GFile(frozen_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="vae")

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
                  num_bins=NUM_BINS)
  test_gen.reset(shuffle=False)

  graph = load_graph(frozen_path)
  for op in graph.get_operations():
    print(op.name)

  x = graph.get_tensor_by_name("vae/x:0")
  training = graph.get_tensor_by_name("vae/training:0")
  z = graph.get_tensor_by_name("vae/sampling/z:0")

  with tf.Session(graph=graph) as sess:
    images = test_gen.get_next_batch()["original_images"]
    emb = sess.run(z, feed_dict={x: images, training: False})
  print(np.shape(emb))
  print(emb)
