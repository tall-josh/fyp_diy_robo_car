
'''
python freeze_graph.py --ckpt-path "./end2end/z_donk/ep_6_loss_3.4e+02_bins_15.ckpt"
                       --graph-path "./end2end/z_donk/model.ckpt.meta"
                       --out-path "./end2end/z_donk/frozne.pb"
                       --outputs "donkey/throttle/Sigmoid" "donkey/steering_prediction"
'''
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
import json
#from show_graph import show_graph
from metrics import Metrics
from freeze_graph import freeze_meta, write_tensor_dict_to_json, load_tensor_names

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=2/11)

INPUTS              = "inputs"
OUTPUTS             = "outputs"
IMAGE_INPUT         = "image_input"
STEERING_PREDICTION = "steering_prediction"
STEERING_PROBS      = "steering_probs"
THROTTLE_PREDICTION = "throttle_prediction"
tensor_dict = {INPUTS  : {IMAGE_INPUT         : ""},
               OUTPUTS : {STEERING_PREDICTION : "",
                          STEERING_PROBS      : "",
                          THROTTLE_PREDICTION : ""}}

'''
intput_or_output = "input" or "output"
key: "descriptive name"
tensor: the tensorflow tensor
'''
def update_tensor_dict(input_or_output, key, tensor):
    tensor_dict[input_or_output][key] = tensor.name

class Model:
    def __init__(self, in_shape, classes, lr=0.001):
        '''
        classes:  List of class names or integers corrisponding to each class being classified
                  by the network. ie: ['left', 'straight', 'right'] or [0, 1, 2]
        '''
        # Define classes
        self.num_bins = len(classes)
        self.classes = np.array(classes, np.float32)
        self.class_lookup = [c for c in classes]

        # Define model
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="input")
        self.y_steering = tf.placeholder(tf.int32,   shape=(None,))
        self.y_throttle = tf.placeholder(tf.float32, shape=(None,))
        self._training = tf.placeholder(tf.bool)
        self.training  = tf.get_variable("training", dtype=tf.bool,
                                          initializer=True, trainable=False)
        self.set_training = self.training.assign(self._training)

        relu    = tf.nn.relu
        sigmoid = tf.nn.sigmoid
        with tf.name_scope("donkey"):
            #            input   num  conv   stride   pad
            conv = conv2d(self.x, 24,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv1")
            conv = conv2d( conv,   32,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv2")
            conv = conv2d( conv,   64,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv3")
            conv = conv2d( conv,   64,  (3,3), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv4")
            conv = conv2d( conv,   64,  (3,3), (1,1),  "same", activation=relu, kernel_initializer=xavier(), name="conv5")
            conv = flatten(conv)
            #             in   num
            conv = dense(  conv, 100, activation=relu, kernel_initializer=xavier(), name="fc1")
            conv = dropout(conv, rate=0.1, training=self.training)

            conv = dense(  conv, 50, activation=relu, kernel_initializer=xavier(), name="fc2")
            conv = dropout(conv, rate=0.1, training=self.training)

            # Steering
            self.logits = dense(conv, self.num_bins, activation=None, kernel_initializer=xavier(), name="logits")
            self.steering_probs = tf.nn.softmax(self.logits, name="steeringi_probs")
            self.steering_prediction = tf.reduce_sum(tf.multiply(self.steering_probs, self.classes),
                                                     axis=1, name="steering_prediction")

            # Throttle
            self.throttle = dense(conv, 1, sigmoid, kernel_initializer=xavier(), name="throttle")

            # keep tensor names for easy freezing/loading later
            update_tensor_dict(INPUTS , IMAGE_INPUT, self.x)
            update_tensor_dict(OUTPUTS, STEERING_PREDICTION, self.steering_prediction)
            update_tensor_dict(OUTPUTS, STEERING_PROBS, self.steering_probs)
            update_tensor_dict(OUTPUTS, THROTTLE_PREDICTION, self.throttle)

        with tf.name_scope("loss"):
            self.loss_steering = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_steering, logits=self.logits)
            self.loss_steering = tf.reduce_mean(self.loss_steering)
            self.loss_throttle = tf.reduce_mean((self.throttle - self.y_throttle)**2)
            self.loss = 0.9*self.loss_steering + 0.001*self.loss_throttle

        tf.summary.scalar("weighted_loss", self.loss)
        tf.summary.scalar("steering_loss", self.loss_steering)
        tf.summary.scalar("throttle_loss", self.loss_throttle)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_step = optimizer.minimize(self.loss)

        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def train(self, train_gen, test_gen, save_dir, epochs=10, restart_ckpt=None):
        return_info = {"graph_path"    : "",
                        "ckpt_path"    : "",
                        "out_path"     : save_dir,
                        "tensor_json"  : ""}
        first_save=True
        best_ckpt = ""
        with tf.Session(config=tf.ConfigProto(gpu_options=GPU_OPTIONS)) as sess:
            merge = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(save_dir+"/logdir/train", sess.graph)
            test_writer  = tf.summary.FileWriter(save_dir+"/logdir/test")

            # Init
            sess.run(self.init_vars)

            # some big number
            best_loss = 10**9

            global_step = 0
            for e in range(epochs):
                sess.run(self.set_training, feed_dict={self._training: True})
                train_gen.reset()
                #t_train = trange(train_gen.steps_per_epoch)
                #t_train.set_description(f"Training Epoch: {e+1}")
                t_train = range(train_gen.steps_per_epoch)
                print(f"Training Epoch: {e+1}")
                for step in t_train:
                    images, steering, throttle = prepare_data(train_gen)
                    _, summary = sess.run([self.train_step, merge],
                                feed_dict={self.x         : images,
                                           self.y_steering: steering,
                                           self.y_throttle: throttle})
                    train_writer.add_summary(summary, global_step)
                    global_step += 1


                sess.run(self.set_training, feed_dict={self._training: False})
                test_gen.reset()
                t_test = range(test_gen.steps_per_epoch)
                print('Testing')
                test_loss = []
                for _ in t_test:
                    images, steering, throttle = prepare_data(test_gen)
                    _loss, summary = sess.run([self.loss, merge],
                                feed_dict={self.x         : images,
                                           self.y_steering: steering,
                                           self.y_throttle: throttle})
                    test_loss.append(_loss)
                    test_writer.add_summary(summary, global_step)
                    global_step += 1

                cur_mean_loss = np.mean(test_loss)
                print("-"*50)
                if cur_mean_loss < best_loss:
                        if first_save:
                            path = f"{save_dir}/graph.meta"
                            self.saver.export_meta_graph(path)
                            return_info["graph_path"] = os.path.abspath(path)

                            path = write_tensor_dict_to_json(save_dir, tensor_dict)
                            return_info["tensor_json"] = os.path.abspath(path)
                            first_save=False
                        best_loss = cur_mean_loss
                        path = f"{save_dir}/ep_{e+1}.ckpt"
                        best_ckpt = self.saver.save(sess, path, write_meta_graph=False)
                        return_info["ckpt_path"] = os.path.abspath(best_ckpt)
                        print("Model saved at {}".format(best_ckpt))

        print(f"Done, final best loss: {best_loss:0.3}")
        return_info["final_loss"] = float(best_loss)
        json.dump(return_info, open(save_dir+"/return_info.json", 'w'))
        return return_info

    def video_drive(self, checkpoint_path, video_path,
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
                    steering_prediction = sess.run(self.steering_prediction,
                                    feed_dict={self.x:img, self.training: False})
                    pwm = self.ExpectedBinToPWM(steering_prediction[0],pwm_min_max)
                    pwm = str(pwm)
                    print("pwm: {}".format(pwm))
                    s.sendall(str.encode(pwm))
                    success, image = vidcap.read()
                    time.sleep(0.10)
            except KeyboardInterrupt:
                    s.close()
                    print("Closed, by user")

# Static methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def prepare_data(generator):
    batch    = generator.get_next_batch()
    images   = batch["images"]
    steering = [ele["steering"] for ele in batch["annotations"]]
    throttle = [ele["throttle"] for ele in batch["annotations"]]
    return images, steering, throttle

