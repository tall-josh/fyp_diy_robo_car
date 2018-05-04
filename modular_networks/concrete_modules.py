import tensorflow as tf
from base_module import BaseModule

class ClassifierModule(BaseModule):

  def prepare_data(self, generator):
    pass

  def make_prediction(self):
    self.probs = tf.nn.softmax(self.logits, name="probs")
    return tf.reduce_sum(tf.multiply(self.probs, self.classes),
                         axis=1, name="prediction")

  def loss_fn(self):
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                              labels=self.y, logits=self.logits)
    return tf.reduce_mean(ce)

class RegressorModule(BaseModule):

  def make_prediction(self):
    return tf.nn.sigmoid(self.logits, name="prediction")

  def loss_fn(self):
    return tf.losses.mean_squared_error(labels=self.y, predictions=self.make_prediction())

class SteeringModule(ClassifierModule):

  def prepare_data(self, generator):
    batch    = generator.get_next_batch()
    images   = batch["images"]
    steering = [ele["steering"] for ele in batch["annotations"]]
    return images, steering

class ThrottleModule(RegressorModule):

  def prepare_data(self, generator):
    batch    = generator.get_next_batch()
    images   = batch["images"]
    steering = [ele["throttle"] for ele in batch["annotations"]]
    return images, steering
