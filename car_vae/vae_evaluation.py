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
      # Init
      # sess.run(self.init_vars)

      generator.reset()
      t_test = trange(generator.steps_per_epoch)
      t_test.set_description(f"Testing Epoch: {e+1}")

       for step in t_test:
        _x, _y = self.prepare_data(generator)

