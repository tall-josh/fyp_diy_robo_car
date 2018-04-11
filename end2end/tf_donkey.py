import time
import socket
import cv2
import tensorflow as tf
from tensorflow.python.layers.core import dense, dropout, flatten
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
from tqdm import trange, tqdm
import numpy as np
import os
import sys
sys.path.append('..')
#from show_graph import show_graph
from metrics import Metrics

class Model:
    def __init__(self, save_dir, in_shape, classes):
        '''
        save_dir: The name given to the network, this will be used in conjunction
                  with 'in_shape' and 'classes' to create a directory to save the
                  model checkpoints.
        in_shape: Shape of the input data being passed to the network, [rows, cols, channels]

        classes:  List of class names or integers corrisponding to each class being classified
                  by the network. ie: ['left', 'straight', 'right'] or [0, 1, 2]
        '''
        print(f"in_shape:           {in_shape}")
        print(f"classes:             {classes}")

        # Make unique directory to save checkpoints
        self.save_dir = save_dir

        # Define classes
        self.num_bins = len(classes)
        self.classes = np.array(classes, np.float32)
        self.class_lookup = [c for c in classes]

        # Define model
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="input")
        self.y = tf.placeholder(tf.int32, shape=(None,), name="label")
        self.training = tf.placeholder(tf.bool, name="training")
        relu    = tf.nn.relu
        with tf.name_scope("encoder"):
            #            input   num  conv   stride   pad
            enc = conv2d(self.x, 24,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="enc1")
            enc = conv2d( enc,   32,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="enc2")
            enc = conv2d( enc,   64,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="enc3")
            enc = conv2d( enc,   64,  (3,3), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="enc4")
            enc = conv2d( enc,   64,  (3,3), (1,1),  "same", activation=relu, kernel_initializer=xavier(), name="enc5")
            enc = flatten(enc)
            #             in   num
            enc = dense(  enc, 100, activation=relu, kernel_initializer=xavier(), name="enc6")
            enc = dropout(enc, rate=0.1, training=self.training)

            enc = dense(  enc, 50, activation=relu, kernel_initializer=xavier(), name="enc7")
            enc = dropout(enc, rate=0.1, training=self.training)

            self.logits = dense(enc, self.num_bins, activation=None, kernel_initializer=xavier(), name="logits")
            self.steering_probs = tf.nn.softmax(self.logits, name="steering")
            self.expected_bin = tf.reduce_sum(tf.multiply(self.steering_probs, self.classes), axis=1)

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="loss")
            self.loss = tf.reduce_mean(self.loss)

        self.saver = tf.train.Saver()

    def Train(self, train_gen, test_gen, save_dir, epochs=10, lr=0.001):

        assert_message = "Name must be unique, This will be the name of the dir we'll used to save checkpoints"
        assert not os.path.exists(save_dir), "{}: {}".format(assert_message, save_dir)
        os.makedirs(save_dir)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(self.loss)

        self.train_loss = []
        self.test_loss  = []
        best_ckpt = ""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best_loss = 10**9  # some big number
            
            
            for e in range(epochs):
                temp_train_loss = []
                train_gen.reset()
                t_train = trange(train_gen.steps_per_epoch)
                t_train.set_description(f"Training Epoch: {e+1}")
                for step in t_train:
                    images, annos = train_gen.get_next_batch()
                    _, loss = sess.run([train_step, self.loss],
                               feed_dict={self.x: images, self.y: annos, self.training: True})
                    temp_train_loss.append(loss)

                self.train_loss.append(np.mean(temp_train_loss))
                test_gen.reset()
                batch_test_loss = []

                t_test = trange(test_gen.steps_per_epoch)
                t_test.set_description('Testing')
                for _ in t_test:
                    images, annos = test_gen.get_next_batch()
                    loss = sess.run(self.loss,
                           feed_dict={self.x: images, self.y: annos, self.training: False})
                    batch_test_loss.append(loss)

                cur_mean_loss = np.mean(batch_test_loss)
                self.test_loss.append(cur_mean_loss)

                print(f"Test Loss: {cur_mean_loss:0.3f}")
                print("-"*50)
                if cur_mean_loss < best_loss:
                        best_loss = cur_mean_loss
                        path = f"{save_dir}/ep_{e+1}_loss_{cur_mean_loss:0.3}_bins_{self.num_bins}.ckpt"
                        best_ckpt = self.saver.save(sess, path)
                        print("Model saved at {}".format(best_ckpt))

        self.SaveLossPlots()
        print(f"Done, final best loss: {best_loss:0.3}")
        return best_ckpt
    
    def SaveLossPlots(self):
        print(f"Train: {self.train_loss}")
        print(f"Test:  {self.test_loss}")
    
    def Evaluate(self, eval_gen, checkpoint_path, save_figs=False, save_dir=None):
        
        if save_figs:
            assert save_dir is not None, "If you want to save the evaluation figs, you'll need to provide a save_dir."
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        eval_gen.reset(shuffle=False)
        metrics = Metrics(self.classes)
        with tf.Session() as sess:
            self.saver.restore(sess,checkpoint_path)

            t_train = trange(eval_gen.steps_per_epoch)
            t_train.set_description("Evaluating your face!")
            for step in t_train:
                images, annos = eval_gen.get_next_batch()
                steering_probs, expected_bin = sess.run([self.steering_probs, self.expected_bin],
                           feed_dict={self.x: images, self.training: False})
                metrics.update(steering_probs, expected_bin, annos)
                
        result = metrics.compute_metrics()
        
        if save_figs:
            metrics.SaveEvalFigs(result, save_dir)
            
        return result

    
    def TrainingResults(self):
        return self.train_loss, self.test_loss, self.test_acc

    def Predict(self, images, checkpoint_path):
        with tf.Session() as sess:
            self.saver.restore(sess,checkpoint_path)
            return sess.run([self.steering_probs, self.expected_bin], feed_dict={self.x:images, self.training: False})

    def ExpectedBinToPWM(self, expected_bin, pwm_min_max=(-0.5, 0.5)):
        pwm_range = abs(pwm_min_max[1] - pwm_min_max[0])
        # between 0 and 1
        norm = expected_bin / self.num_bins
        # between -0.5 and 0.5
        zero_cent = norm - 0.5
        return zero_cent * pwm_range

    def ExpectedBinToDeg(self, expected_bin, steering_range_deg=40):
        # between 0 and 1
        norm = expected_bin / self.num_bins
        # between -0.5 and 0.5
        zero_cent = norm - 0.5
        return zero_cent * steering_range_deg

    def GetGraph(self):
        return tf.get_default_graph().as_graph_def()

    def VideoDrive(self, checkpoint_path, video_path,
                   car_ip, pwm_min_max=(-0.5, 0.5), port=5555, steering_range_deg=40):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((car_ip, port))
        with tf.Session() as sess:
            self.saver.restore(sess,checkpoint_path)
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            if not success:
                print("Sad times :-(")
                return None

            try:
                while success:
                    image = image[40:,:]
                    orig_h      = image.shape[0]
                    orig_w      = image.shape[1]
                    x_scale     = 160. / 640.
                    y_scale     = 120. / 320.
                    img = cv2.resize(image, None, fx=x_scale, fy=y_scale)
                    img = np.expand_dims(img, axis=0)
                    expected_bin = sess.run(self.expected_bin,
                                    feed_dict={self.x:img, self.training: False})
                    pwm = self.ExpectedBinToPWM(expected_bin[0],pwm_min_max)
                    pwm = str(pwm)
                    print("pwm: {}".format(pwm))
                    s.sendall(str.encode(pwm))
                    success, image = vidcap.read()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                    s.close()
                    print("Closed, by user")
