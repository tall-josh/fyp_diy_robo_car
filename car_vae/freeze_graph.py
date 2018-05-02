# NOTE:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
import tensorflow as tf
import os

load_dir   = "z_freeze"
graph_path = os.path.join(load_dir, "model_def.meta")
ckpt_path  = os.path.join(load_dir, "ep_2_loss_14245.817.ckpt")
graph = None
with tf.Session(graph=tf.Graph()) as sess:
  saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
  sess.run( tf.global_variables_initializer())
  saver.restore(sess, ckpt_path)
  #graph = tf.get_default_graph()
  output_graph_def = tf.graph_util.convert_variables_to_constants(
                       sess,
                       tf.get_default_graph().as_graph_def(),
                       ["sampling/z", "beta"])
  outpath = os.path.join(load_dir, "FROZEN.pb")
  with tf.gfile.GFile(outpath, "wb") as f:
    f.write(output_graph_def.SerializeToString())

  print("%d ops in the final graph." % len(output_graph_def.node))
