# NOTE:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
import tensorflow as tf
import os
import json

INPUTS              = "inputs"
OUTPUTS             = "outputs"
'''
To use this module you'll need to define a dict with a the structure shown below.

!!! 'INPUTS' and 'OUTPUTS' are non negotiable !!!

<IMAGE_INPUT> and <PREDICTION> are examples and my 
be changed to suit the model.

INPUTS              = "inputs"
OUTPUTS             = "outputs"
IMAGE_INPUT         = "image_input"
PREDICTION          = "prediction"
tensor_dict = {INPUTS  : {IMAGE_INPUT : ""},
               OUTPUTS : {PREDICTION  : ""}}
               

(1)
Use update_tensor_dict to write to the tensor_dict as you
build your model. This will keep track of the tensor names.

params:
intput_or_output = "input" or "output"
key: "descriptive name"
tensor: the tensorflow tensor
'''
def update_tensor_dict(input_or_output, key, tensor):
    tensor_dict[input_or_output][key] = tensor.name
'''
(2)
When your model is build, use this function to save
the dict to a json for easy recovery at a later time.
'''
def write_tensor_dict_to_json(save_dir, tensor_dict):
    path = os.path.join(save_dir, "tensor_names.json")
    print(f"tensor_dict: {tensor_dict}")
    with open(path, 'w') as f:
        json.dump(tensor_dict, f)
    print(f"tensor dict saved at {path}")
    return os.path.abspath(path)

'''
(3)
Once training is complete you can use this to freeze your
model at a desire checkpoint.

params:
graph_path: path to graph def (usually '.meta')
ckpt_path:  path to checkpoint of model (usually '.ckpt')
out_path: /where/do/you/want/to/save/to/frozen.pb
tensor_names_json: the path to the json writen in step (2)
'''
def freeze_meta(graph_path, ckpt_path, out_path, tensor_names_json):
    with open(tensor_names_json, 'r') as f:
        tensor_names = json.load(f)
    print(f"tensor_names: {tensor_names}")
    # remove :0 from tensor names. Freezing need this for some reason
    output_names = [tensor_names[OUTPUTS][key].split(":")[0] for key in tensor_names[OUTPUTS].keys()]
    print(f"output_names: {output_names}")
    path = freeze_graph(graph_path, ckpt_path, out_path, output_names)
    return os.path.abspath(path)

def load_tensor_names(tensor_name_json):
    with open(tensor_name_json, 'r') as f:
        tensor_names = json.load(f)
    print(f"tensor names loaded:\n{tensor_names.keys()}")
    return tensor_names


'''
example usage:

  python freeze_graph.py --ckpt-path  path/to/some_checkpoint.ckpt
                         --graph-path path/to/corrisponding_graph.meta
                         --out-path   save/here/please.pb
                         --outputs    'name_scope/tensor_name1' 'op_name42'

  --outputs: Path to json containing tensor names
             A list of string corrisponding to the output tensors/ops defined
             in the model, the/slash/notation indicates a tensor resides
             within a tf.name_scope.
'''
def freeze_graph(graph_path, ckpt_path, out_path, output_names):
  graph = None
  with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
    sess.run( tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
 #   for op in sess.graph.get_operations():
 #     print(op)
    #graph = tf.get_default_graph()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                         sess,
                         tf.get_default_graph().as_graph_def(),
                         output_names)
    with tf.gfile.GFile(out_path, "wb") as f:
      f.write(output_graph_def.SerializeToString())

  print("%d ops in the final graph." % len(output_graph_def.node))
  print("FROZEN graph at: {}".format(out_path))
  return out_path


if __name__ == "__main__":
  import argparse as argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt-path',  type=str, required=True)
  parser.add_argument('--graph-path', type=str, required=True)
  parser.add_argument('--out-path',   type=str, required=True)
  parser.add_argument('--outputs',    type=str, required=False, nargs="+",
                      help="name of output tensors")
  parser.add_argument('--tensor-json',    type=str, required=False,
                      help="json with tensor names")
  args = parser.parse_args()
  graph_path   = args.graph_path
  ckpt_path    = args.ckpt_path
  out_path     = args.out_path
  output_names = args.outputs

  if args.tensor_json is not None:
    tensor_names = load_tensor_names(args.tenson_json)
    output_names = [n.split(":")[0] for n in tensor_names[OUTPUT_NAMES]]

  freeze_graph(graph_path, ckpt_path, out_path, output_names)
