# NOTE:
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

from vae_data_generator import DataGenerator
import tensorflow as tf
from utils import *
from tqdm import trange
import sys
sys.path.append('../')

train = load_dataset("../data/evened_train.txt")
test = load_dataset("../data/evened_test.txt")
image_dir = "../data/clr_120_160/images"
anno_dir  = "../data/clr_120_160/annotations"
batch_size  = 50
NUM_BINS = 15

train_gen   = DataGenerator(batch_size=batch_size,
                  data_set=train[:200],
                  image_dir=image_dir,
                  anno_dir=anno_dir,
                  num_bins=NUM_BINS)

test_gen    = DataGenerator(batch_size=batch_size,
                  data_set=test[:50],
                  image_dir=image_dir,
                  anno_dir=anno_dir,
                  num_bins=NUM_BINS)

sample_gen  = DataGenerator(batch_size=10,
                  data_set=test[:10],
                  image_dir=image_dir,
                  anno_dir=anno_dir,
                  num_bins=NUM_BINS,
                  shuffle=False)

graph = None
with tf.Session() as sess:
  load_dir   = "z_freeze"
  graph_path = os.path.join(load_dir, "model_def.meta")
  ckpt_path  = os.path.join(load_dir, "ep_2_loss_14245.817.ckpt")
  new_saver = tf.train.import_meta_graph(graph_path)
  sess.run( tf.global_variables_initializer())
  new_saver.restore(sess, ckpt_path)
  graph = tf.get_default_graph()

#  ops = [x for x in tf.get_default_graph().get_operations()]
#  for o in ops:
#    print(o)

  train_step = graph.get_operation_by_name("train_step")
  loss       = graph.get_operation_by_name("loss/total_loss")
  x = graph.get_tensor_by_name("x:0")
  y = graph.get_tensor_by_name("y:0")
  z        = graph.get_operation_by_name("sampling/z")
  training = graph.get_tensor_by_name("training:0")
  beta     = graph.get_tensor_by_name("beta:0")

  epochs = 5
  count = 1.
  for e in range(epochs):
    # Tensorboard
#    merge = tf.summary.merge_all()
    # Begin Training
    train_gen.reset()
    t_train = trange(train_gen.steps_per_epoch)
    t_train.set_description(f"Training Epoch: {e+1}")
    for step in t_train:

      batch = train_gen.get_next_batch()
      _, _loss, _beta = sess.run([train_step, loss, beta],
                 feed_dict={x: batch["augmented_images"],
                              y: batch["original_images"],
                              training: True})
      print("BETA: {}".format(_beta))
      count += 1

#    # Begin Testing
#    embeddings      = []
#    test_gen.reset(shuffle=False)
#    t_test = trange(test_gen.steps_per_epoch)
#    t_test.set_description('Testing')
#    for _ in t_test:
#      batch = test_gen.get_next_batch()
#      loss, zeds = sess.run([loss, z],
#                 feed_dict={x: batch["augmented_images"],
#                            y: batch["original_images"],
#                            beta: _beta,
#                            training: False})
#      embeddings.extend(zeds)

