# NOTE:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
import tensorflow as tf
import os
'''
example usage:

  python freeze_graph.py --ckpt-path  path/to/some_checkpoint.ckpt
                         --graph-path path/to/corrisponding_graph.meta
                         --out-path   save/here/please.pb
                         --outputs    'name_scope/tensor_name1' 'op_name42'

  --outputs: A list of string corrisponding to the output tensors/ops defined
             in the model, the/slash/notation indicates a tensor resides
             within a tf.name_scope.
'''
def freeze_graph(graph_path, ckpt_path, out_path, output_names):

  graph = None
  with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
    sess.run( tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
    #graph = tf.get_default_graph()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                         sess,
                         tf.get_default_graph().as_graph_def(),
                         output_names)
    with tf.gfile.GFile(out_path, "wb") as f:
      f.write(output_graph_def.SerializeToString())

  print("%d ops in the final graph." % len(output_graph_def.node))
  print("FROZEN graph at: {}".format(out_path))

if __name__ == "__main__":
  import argparse as argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt-path',  type=str, required=True)
  parser.add_argument('--graph-path', type=str, required=True)
  parser.add_argument('--out-path',   type=str, required=True)
  parser.add_argument('--outputs',    type=str, required=True, nargs="+",
                      help="name of output tensors")
  args = parser.parse_args()
  graph_path   = args.graph_path
  ckpt_path    = args.ckpt_path
  out_path     = args.out_path
  output_names = args.outputs
  freeze_graph(graph_path, ckpt_path, out_path, output_names)
