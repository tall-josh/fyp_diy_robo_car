import time
import socket
import cv2
import tensorflow as tf
from tensorflow.python.layers.core import dense, dropout, flatten
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
from tqdm import trange, tqdm
import numpy as np
import sys
sys.path.append('..')
from show_graph import show_graph
from metrics import Metrics

class Model:
    def __init__(self, name, in_shape, classes):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=(None,)+in_shape, name="input")
        self.y = tf.placeholder(tf.int32, shape=(None,), name="label")
        self.training = tf.placeholder(tf.bool, name="training")
        self.num_bins = len(classes)
        #self.classes = tf.constant(classes, dtype=tf.float32, name="classes")
        self.classes = np.array(classes, np.float32)
        self.name = "{}_bins_{}".format(name, len(classes))

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

            self.logits = dense(enc, 15, activation=None, kernel_initializer=xavier(), name="logits")
            self.steering = tf.nn.softmax(self.logits, name="steering")

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="loss")
            self.loss = tf.reduce_mean(self.loss)
        
        self.saver = tf.train.Saver()
#        with tf.name_scope("accuracy"):
#            '''
#            eg: This is not code, it's more like a derivation
#                just for my own reference.
#            classes    = [0,1,2]  # index of steering angle bins
#            steering   = [[0.1, 0.7, 0.2], [0.8, 0.2, 0.0]]
#            true       = [1, 0]
#
#            Expected value of the pdf output by the softmax opperation
#            prediction = [[0.0, 0.7, 0.4], [0.0, 0.2, 0.0]] # classes * steering
#            prediction = [1.1, 0.2] # tf.reduce_sum axis=1
#
#            abs_dif    = [0.1, 0.2]  # abs(true - prediction)
#            percent_er = [0.1/3, 0.2/3] # where 3 is the number of classes
#            acc        = 1 - pervent_er
#            mean_acc   = tf.reduce_mean(acc)
#            '''
#            self.prediction = tf.reduce_sum(tf.multiply(self.steering, self.classes), axis=1)
#            abs_diff   = tf.abs(self.prediction - tf.cast(self.y, tf.float32))
#            percent_error = abs_diff / tf.cast(tf.shape(self.classes), tf.float32)
#            self.accuracy   = 1. - percent_error
#            self.mean_accuracy = tf.reduce_mean(self.accuracy)

        


    def Train(self, train_gen=None, test_gen=None, epochs=10, lr=0.001):
        if train_gen is None or test_gen is None:
            print("This is a lovely message from your Model object's Train function.")
            print("How can I train if you don't give me data ya ding bat!")
            return None

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(self.loss)

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

                t_train = trange(train_gen.steps_per_epoch)
                t_train.set_description(f"Epoch {e+1}")
                for step in t_train:
                    images, annos = train_gen.get_next_batch()
                    _, loss = sess.run([train_step, self.loss], \
                               feed_dict={self.x: images, self.y: annos, self.training: True})
                    self.train_loss.append(self.loss)
                    t_train.set_description("Train Loss: {:.3}".format(loss))

                test_gen.reset()
                cur_test_loss = []
                cur_test_acc  = []

                t_test = trange(test_gen.steps_per_epoch)
                t_test.set_description('Testing')
                for _ in t_test:
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
                        path = "{}_ep_{}_loss_{:.3}.ckpt".format(self.name, e+1, cur_mean_loss)
                        saved_path = self.saver.save(sess, path)
                        print("Model saved at {}".format(saved_path))

        print("Done, final best loss: {:.3}".format(best_loss))

    def Evaluate(self, eval_gen, checkpoint_path):
        eval_gen.reset()
        print(f"classes: {np.shape(self.classes)[0]}")
        metrics = Metrics(self.classes)
        with tf.Session() as sess:
            self.saver.restore(sess,checkpoint_path)

            t_train = trange(eval_gen.steps_per_epoch)
            t_train.set_description("Evaluating your face!")
            for step in t_train:
                images, annos = eval_gen.get_next_batch()
                steering_probs = sess.run(self.steering, \
                           feed_dict={self.x: images, self.training: False})
                metrics.update(steering_probs, annos)

        return metrics.compute_metrics()


    def TrainingResults(self):
        return self.train_loss, self.test_loss, self.test_acc

    def Predict(self, images, checkpoint_path):
        with tf.Session() as sess:
            self.saver.restore(sess,checkpoint_path)
        #     self.saver.restore(sess,'./ep_10-step_3800-loss_1.2003761529922485.ckpt')
            return sess.run([self.steering, self.prediction], feed_dict={self.x:images, self.training: False})

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
#'./ep_19-step_50-loss_0.5418351888656616.ckpt'

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
                    pred = sess.run(self.prediction,
                                    feed_dict={self.x:img, self.training: False})
                    pwm = self.ExpectedBinToPWM(pred,pwm_min_max)
                    pwm = str(pwm[0])
                    print("pwm: {}".format(pwm))
                    s.sendall(str.encode(pwm))
                    success, image = vidcap.read()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                    s.close()
                    print("Closed, by user")


def VideoDrive(checkpoint_path, video_path, num_bins=15, in_shape=(120,160,3),
               car_ip="192.168.43.56", pwm_min_max=(-0.5, 0.5), port=5555, steering_range_deg=40):

    classes = [x for x in range(num_bins)]
    car_brain = Model("test_name", in_shape=in_shape, classes=classes)
    car_brain.VideoDrive(checkpoint_path, video_path, car_ip)

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    print("Hmmm: {}".format(args.ckpt))
    ckpt = "ep_18-step_161-loss_1.05.ckpt"
    video_path = "/home/jp/Documents/FYP/ml/data/videoplayback.mp4"
    VideoDrive(ckpt, video_path)

if __name__ == "__main__":
    main()
