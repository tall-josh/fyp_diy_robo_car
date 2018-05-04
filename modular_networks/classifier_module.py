import tensorflow as tf
from modular_network import Module

class ClassifierModule(Module):

  def prediction(self):
    self.probs = tf.nn.softmax(self.logits, name="probs")
    return tf.reduce_sum(tf.multiply(self.probs, self.classes),
                         axis=1, name="prediction")

  def loss_fn(self):
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                              labels=self.y, logits=self.logits)
    return tf.reduce_mean(ce)
