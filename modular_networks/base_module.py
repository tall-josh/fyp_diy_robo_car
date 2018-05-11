# NOTE:
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

from utils import *
from tqdm import trange
import tensorflow as tf
from tensorflow.python.layers.core import dense, dropout, flatten
from tensorflow.python.layers.convolutional import conv2d, conv2d_transpose
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
from tensorflow.contrib.tensorboard.plugins import projector
from load_frozen import load_graph
from freeze_graph import freeze_meta, write_tensor_dict_to_json, load_tensor_names
INPUTS              = "inputs"
OUTPUTS             = "outputs"
IMAGE_INPUT         = "image_input"
PREDICTION          = "prediction"
tensor_dict = {INPUTS  : {IMAGE_INPUT : ""},
               OUTPUTS : {PREDICTION  : ""}}

GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=2/11)

def update_tensor_dict(input_or_output, key, tensor):
  tensor_dict[input_or_output][key] = tensor.name

class BaseModule(object):

  def __init__(self, encoder, layer_def, classes, lr=0.001):
    """
    encoder:
      {"path": string,      # path to frozen encoder pb file
       "input_tensor_name":  string,
       "output_tensor_name": string,
       "name": string       # name of encoder
      }

    layer_def: Node definition for fully connected network. List of dict.
      [{"neurons"   : int,         # number of neurons
        "activation": function,    # activation  function
        "init"      : function,    # tf initialization function
        "name"      : string,      # used to identify tensor
        "dropout"   : float(0-1)   # if 1 then no dropout is added. keep_prob=1
       },
       {...}
      ]
    classes: list of classes the network will be identifying, can be list of int
             of list of string.
     """
    self.classes    = classes
    self.num_bins   = len(classes)
    self.class_idxs = np.array(range(len(classes)), np.float32)

    # Load frozen encoder
    self.encoder = load_graph(encoder["path"], encoder["name"])
    with self.encoder.as_default() as graph:
      # images into the network
      self.x = self.encoder.get_tensor_by_name(encoder["name"] + "/" + \
                                                     encoder["input_tensor_name"])
      # embedding from encoder
      _z     = self.encoder.get_tensor_by_name(encoder["name"] + "/" + \
                                                  encoder["output_tensor_name"])
      # ensure encoder's weights are held static
      self.z = tf.stop_gradient(_z)

      # known label
      self.y = tf.placeholder(tf.int32, name="y")

      self._training    = tf.placeholder(tf.bool)
      self.training     = tf.get_variable("training", dtype=tf.bool,
                                           initializer=True,
                                           trainable=False)
      self.set_training = self.training.assign(self._training)
      lay1 = dense(self.z, 40, activation=tf.nn.relu, kernel_initializer=xavier())
      lay1 = dropout(lay1, rate=0.5)
      lay1 = dense(lay1, 30, activation=tf.nn.relu, kernel_initializer=xavier())
      lay1 = dropout(lay1, rate=0.5)
      lay1 = dense(lay1, 20, activation=tf.nn.relu, kernel_initializer=xavier())
      lay1 = dropout(lay1, rate=0.5)
      lay1 = dense(lay1, 15, activation=None, kernel_initializer=xavier())

      self.logits   = lay1
      self.prediction = self.make_prediction()

      self.loss     = self.loss_fn()

      # Tensorboard
      tf.summary.scalar("loss", self.loss)

      optimizer       = tf.train.AdamOptimizer(learning_rate=lr)
      self.train_step = optimizer.minimize(self.loss)
      self.init_vars  = tf.global_variables_initializer()

      update_tensor_dict(INPUTS,  IMAGE_INPUT, self.x)
      update_tensor_dict(OUTPUTS, PREDICTION, self.prediction)

      self.saver = tf.train.Saver()
      self.graph = graph

  def train(self, train_gen, test_gen, save_dir, epochs):
    return_info = {"graph_path"    : "",
                   "ckpt_path"    : "",
                   "out_path"     : save_dir,
                   "tensor_json"  : ""}
    with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=GPU_OPTIONS)) as sess:
      # Tensorboard
      merge = tf.summary.merge_all()

      train_writer = tf.summary.FileWriter(save_dir+"/logdir/train", sess.graph)
      test_writer  = tf.summary.FileWriter(save_dir+"/logdir/test")

      # some big number
      best_loss = 10**9.0

      # Init
      sess.run(self.init_vars)

      global_step = 0
      first_save = True
      for e in range(epochs):

        # Begin Training
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sess.run(self.set_training, feed_dict={self._training: True})
        train_gen.reset()
        t_train = trange(train_gen.steps_per_epoch)
        t_train.set_description(f"Training Epoch: {e+1}")

        for step in t_train:
          _x, _y = self.prepare_data(train_gen)

          _, loss, summary = sess.run([self.train_step,
                                       self.loss, merge],
                                       feed_dict={self.x: _x,
                                                  self.y: _y})
          # Tensorboard
          train_writer.add_summary(summary, global_step)
          global_step += 1
          t_train.set_description(f"{np.mean(loss):.3f}")
          # Begin Testing
          # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          if global_step%50 == 0:
            print(f"Testing Epoch: {e+1}")
            sess.run(self.set_training, feed_dict={self._training: False})
            test_gen.reset()
            _x, _y = self.prepare_data(test_gen)

            loss, summary = sess.run([self.loss, merge],
                                   feed_dict={self.x: _x,
                                              self.y: _y})
            cur_loss = np.mean(loss)
            print(f"Test Loss: {cur_loss:.3f}")
            if cur_loss < best_loss:
              best_loss = cur_loss
              path = f"{save_dir}/ep_{e+1}.ckpt"
              best_ckpt = self.saver.save(sess, path, write_meta_graph=False)
              return_info["ckpt_path"]=os.path.abspath(best_ckpt)
              print("Model saved at {}".format(best_ckpt))

              if first_save:
                path = os.path.join(save_dir, "graph.meta")
                self.saver.export_meta_graph(path)
                return_info["graph_path"] = os.path.abspath(path)

                path = write_tensor_dict_to_json(save_dir, tensor_dict)
                return_info["tensor_json"] = os.path.abspath(path)
                first_save=False

              # Tensorboard
              test_writer.add_summary(summary, global_step)

      print(f"Done, final best loss: {best_loss:0.3f}")
      return_info["final_loss"] = float(best_loss)
      json.dump(return_info, open(save_dir+"/return_info.json", 'w'))
      return return_info

  def prepare_data(self):
    pass

  def make_prediction(self):
    pass

  def loss_fn(self):
    pass

  def update_tensor_dict(input_or_output, key, tensor):
    pass
