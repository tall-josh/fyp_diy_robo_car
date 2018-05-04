# NOTE:
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

from utils import *
from tqdm import trange
import tensorflow as tf
from load_frozen import load_graph

class VaeVisualization(object):

  def __init__(self, frozen_vae):

    # Load frozen vae
    self.vae = load_graph(frozen_vae["path"], frozen_vae["name"])
    with self.vae.as_default() as graph:
      # images into the network
      self.x = self.vae.get_tensor_by_name(vae["name"] + "/" + \
                                                     vae["input_tensor_name"])
      # embedding from encoder
      _z = self.vae.get_tensor_by_name(vae["name"] + "/" + \
                                                  vae["z_tensor_name"])
      # ensure encoder's weights are held static
      self.z = tf.stop_gradient(_z)

      # reconstruction from decoder
      _y = self.vae.get_tensor_by_name(vae["name"] + "/" + \
                                                  vae["decoder_tensor_name"])
      # ensure decoder's weights are held static
      self.y = tf.stop_gradient(_y)

      self.graph = graph

  def visualize_some_cool_shit(self, generator):

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

