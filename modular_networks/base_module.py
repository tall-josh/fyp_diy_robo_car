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

      # Setup metwork that will take [n,50] embedding vectors.
      _ = self.z
      for layer in layer_def:
        _ = dense(_,
                  units              = layer["neurons"],
                  activation         = layer["activation"],
                  kernel_initializer = layer["init"],
                  name               = layer["name"])
        drop = layer["dropout"]
        if drop!= 1.:
          _ = dropout(_, rate=drop, training=self.training)

      self.logits   = _
      self.prediction = self.make_prediction()

      self.loss     = self.loss_fn()

      # Tensorboard
      tf.summary.scalar("loss", self.loss)

      optimizer       = tf.train.AdamOptimizer(learning_rate=lr)
      self.train_step = optimizer.minimize(self.loss)
      self.init_vars  = tf.global_variables_initializer()

      self.saver = tf.train.Saver()
      self.graph = graph

  def train(self, train_gen, test_gen, save_dir, epochs):

    with tf.Session(graph=self.graph) as sess:
      # Tensorboard
      merge = tf.summary.merge_all()

      train_writer = tf.summary.FileWriter(save_dir+"/logdir/train", sess.graph)
      test_writer  = tf.summary.FileWriter(save_dir+"/logdir/test")

      # some big number
      best_loss = 10**9.0

      # Init
      sess.run(self.init_vars)

      global_step = 0
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

        # Begin Testing
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sess.run(self.set_training, feed_dict={self._training: False})
        test_gen.reset()
        t_test = trange(test_gen.steps_per_epoch)
        t_test.set_description(f"Testing Epoch: {e+1}")

        for step in t_test:
          _x, _y = self.prepare_data(test_gen)

          _, summary = sess.run([self.loss, merge],
                                 feed_dict={self.x: _x,
                                            self.y: _y})
          # Tensorboard
          test_writer.add_summary(summary, global_step)
          global_step += 1

  def prepare_data(self):
    pass
  def make_prediction(self):
    pass

  def loss_fn(self):
    pass

