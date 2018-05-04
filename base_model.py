class BaseModel():

  def __init__(self, in_shape):
    tf.reset_default_graph
    self.x = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="input")
    self.y = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="label")
    self.training   = tf.placeholder(tf.bool, name="training")

  def Train(self, train_gen, test_gen, save_dir, epochs=10, lr=0.001, sample_inf_gen=None):

    return best_ckpt
