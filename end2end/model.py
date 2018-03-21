import tensorflow as tf
from tensorflow.contrib import layers
from tqdm import trange, tqdm
import numpy as np
from show_graph import show_graph

class Model:
    def __init__(self, num_bins=8):
        tf.reset_default_graph()
        self.num_bins = num_bins
        self.x = tf.placeholder(tf.float32, shape=(None, 80,80,1), name="input")
        self.y = tf.placeholder(tf.int32, shape=(None,), name="label")
        self.training = tf.placeholder(tf.bool, name="training")
        # used for calculating average of predicted distribution 
        self.classes = tf.constant(list(range(0,self.num_bins)), dtype=tf.float32, name="classes")

        with tf.name_scope("lay1"):
            filt_num    = 10
            filt_size   = (8,8)
            filt_stride = (2, 2)  # (Height, Width)
            padding     = "same"
            activation  = tf.nn.relu
            # xavier initializer by default :-)
            self.lay1 = layers.conv2d(self.x,   filt_num, filt_size, filt_stride, padding, activation_fn=activation, scope="lay1")

        with tf.name_scope("lay2"):
            filt_num    = 20
            filt_size   = (4,4)
            filt_stride = (2, 2)
            padding     = "same"
            activation  = tf.nn.relu

            self.lay2 = layers.conv2d(self.lay1,   filt_num, filt_size, filt_stride, padding, activation_fn=activation, scope="lay2")

        with tf.name_scope("lay3"):
            filt_num    = 40
            filt_size   = (2,2)
            filt_stride = (1, 1)
            padding     = "same"
            activation  = tf.nn.relu

            self.lay3 = layers.conv2d(self.lay2,   filt_num, filt_size, filt_stride, padding, activation_fn=activation, scope="lay3")
            self.lay3_flat = layers.flatten(self.lay3)

        with tf.name_scope("fc1"):
            neurons = 1600
            activation  = tf.nn.relu
            dropout_rate = 0.6

            self.fc1 = layers.fully_connected(self.lay3_flat, neurons, activation_fn=activation, scope="fc1")
            self.fc1_drop = tf.layers.dropout(self.fc1, rate=dropout_rate, training=self.training, name="drop1")

        with tf.name_scope("fc2"):
            neurons = 160
            activation  = tf.nn.relu
            dropout_rate = 0.6

            self.fc2 = layers.fully_connected(self.fc1_drop, neurons, activation_fn=activation, scope="fc2")
            self.fc2_drop = tf.layers.dropout(self.fc2, rate=dropout_rate, training=self.training, name="drop2")

        with tf.name_scope("output"):
            neurons = self.num_bins
            activation  = None

            self.logits = layers.fully_connected(self.fc2_drop, neurons, activation_fn=activation, scope="lay_logits")
            self.probs = layers.softmax(self.logits, scope="probs")

        with tf.name_scope("loss"):
            self.per_class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="class_loss")
            self.loss = tf.reduce_mean(self.per_class_loss)

        with tf.name_scope("accuracy"):
            '''
            eg: This is not code, it's more like a derivation
                just for my own reference.
            classes    = [0,1,2]  # index of steering angle bins
            probs      = [[0.1, 0.7, 0.2], [0.8, 0.2, 0.0]]
            true       = [1, 0]
            
            Expected value of the pdf output by the softmax opperation
            prediction = [[0.0, 0.7, 0.4], [0.0, 0.2, 0.0]] # classes * probs
            prediction = [1.1, 0.2] # tf.reduce_sum axis=1 
            
            abs_dif    = [0.1, 0.2]  # abs(true - prediction)
            percent_er = [0.1/3, 0.2/3] # where 3 is the number of classes
            acc        = 1 - pervent_er
            mean_acc   = tf.reduce_mean(acc)
            '''
            self.prediction = tf.reduce_sum(tf.multiply(self.probs, self.classes), axis=1)
            abs_diff   = tf.abs(self.prediction - tf.cast(self.y, tf.float32))
            percent_error = abs_diff / tf.cast(tf.shape(self.classes), tf.float32)
            self.accuracy   = 1. - percent_error
            self.mean_accuracy = tf.reduce_mean(self.accuracy)
    
    
    def Train(self, train_gen=None, test_gen=None, epochs=10, lr=0.001):
        if train_gen is None or test_gen is None:
            print("This is a lovely message from your Model object's Train function.")
            print("How can I train if you don't give me data ya ding bat!")
            return None
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        
        self.train_loss = list()
        self.test_loss  = list()
        self.test_acc   = list()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best_loss = 10**9  # some big number
            for e in range(epochs):
                
                train_gen.reset()
                print("Epoch {}".format(e+1))
                print("Training")
                
                pbar = tqdm(list(range(train_gen.steps_per_epoch)))
                for step in pbar:
                    images, annos = train_gen.get_next_batch()
                    _, loss = sess.run([train_step, self.loss], \
                               feed_dict={self.x: images, self.y: annos, self.training: True})
                    self.train_loss.append(loss)
                    pbar.set_description("Train Loss: {:.3}".format(loss))
                
                test_gen.reset()
                cur_test_loss = []
                cur_test_acc  = []
                print("Testing")
                pbar = tqdm(list(range(test_gen.steps_per_epoch)))
                for _ in trange(test_gen.steps_per_epoch):
                    images, annos = test_gen.get_next_batch()
                    loss, acc = sess.run([self.loss, self.mean_accuracy], \
                           feed_dict={self.x: images, self.y: annos, self.training: False})
                    cur_test_loss.append(loss)
                    cur_test_acc.append(acc)
                    
                cur_mean_loss = np.mean(cur_test_loss)
                cur_mean_acc  = np.mean(cur_test_acc)
                self.test_loss.append(cur_mean_loss)
                self.test_acc.append( cur_mean_acc)
                
                print("Test Loss: {:.3f}, Test Acc: {:.3f}".format(cur_mean_loss, cur_mean_acc))
                print("-"*50)
                if cur_mean_loss < best_loss:
                        best_loss = cur_mean_loss
                        saved_path = self.saver.save(sess, "./ep_{}-step_{}-loss_{:.3}.ckpt".format(e+1, step,cur_mean_loss))
                        print("Model saved at {}".format(saved_path))

        print("Done, final best loss: {:.3}".format(best_loss))

    def TrainingResults(self):
        return self.train_loss, self.test_loss, self.test_acc
    
    def Predict(self, images, checkpoint_path):
        with tf.Session() as sess:
            self.saver.restore(sess,checkpoint_path)
        #     self.saver.restore(sess,'./ep_10-step_3800-loss_1.2003761529922485.ckpt')
            return sess.run([self.probs, self.prediction], feed_dict={self.x:images, self.training: False})
       
    
#    def Predict_Batch(self, images, checkpoint_path):
#        preds = list()
#        with tf.Session() as sess:
#            self.saver.restore(sess,checkpoint_path)
        #     saver.restore(sess,'./ep_10-step_3800-loss_1.2003761529922485.ckpt')
        
#            preds.append(sess.run(self.prediction,\
#                                feed_dict={self.x:images, self.training: False}))
#        return preds
    
    def ExpectedBinToDeg(self, expected_bin, steering_range_deg=80):
        # between 0 and 1
        norm = expected_bin / self.num_bins
        # between -0.5 and 0.5
        zero_cent = norm - 0.5
        return zero_cent * steering_range_deg
        
    def GetGraph(self):
        return tf.get_default_graph().as_graph_def()
#'./ep_19-step_50-loss_0.5418351888656616.ckpt'