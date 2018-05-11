import cv2
import tensorflow as tf
from tensorflow.python.layers.core import dense, dropout, flatten
from tensorflow.python.layers.convolutional import conv2d, conv2d_transpose
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import trange, tqdm
import numpy as np
import os
#from show_graph import show_graph
import json
from metrics import Metrics
from freeze_graph import freeze_meta, write_tensor_dict_to_json, load_tensor_names
import sys
sys.path.append('..')
from utils import save_images

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=5/11)

INPUTS              = "inputs"
OUTPUTS             = "outputs"
IMAGE_INPUT         = "image_input"
EMBEDDING           = "embedding"
RECONSTRUCTION      = "reconstruction"
tensor_dict = {INPUTS  : {IMAGE_INPUT    : ""},
               OUTPUTS : {EMBEDDING      : "",
                          RECONSTRUCTION : ""}}
'''
intput_or_output = "input" or "output"
key: "descriptive name"
tensor: the tensorflow tensor
'''
def update_tensor_dict(input_or_output, key, tensor):
    tensor_dict[input_or_output][key] = tensor.name

class Model:
    def __init__(self, in_shape, lr=0.001, embedding_dim=50, num_projections=20):
        '''
        classes:  List of class names or integers corrisponding to each class being classified
                  by the network. ie: ['left', 'straight', 'right'] or [0, 1, 2]
        '''
        # number of elements in the embedding vector
        self.embedding_dim = embedding_dim
        # number of test embeddings to project in tensorboard
        self.num_projections = num_projections
        # Define model
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="x")
        self.y = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="y")

        self._training    = tf.placeholder(tf.bool)
        self.training     = tf.get_variable("training", dtype=tf.bool,
                                             initializer=True,
                                             trainable=False)
        self.set_training = self.training.assign(self._training)

        self._beta        = tf.placeholder(dtype=tf.float32)
        self.beta         = tf.get_variable("beta", dtype=tf.float32,
                                             initializer=0.,
                                             trainable=False)
        self.update_beta  = self.beta.assign(self._beta)

        self._embeddings  = tf.placeholder(tf.float32,
                                           shape=[None, self.embedding_dim])
        self.embeddings   = tf.get_variable("embeddings",
                                             dtype=tf.float32,
                                             shape=
                                            [self.num_projections,
                                             self.embedding_dim],
                                             initializer=tf.zeros_initializer(),
                                             trainable=False)
        self.update_embeddings = self.embeddings.assign(self._embeddings)

        paddings = tf.constant([[0,0],[4,4],[0,0],[0,0]])
        relu    = tf.nn.relu
        sigmoid = tf.nn.sigmoid
        with tf.name_scope("encoder"):
            # Padding invector so reconstruction returns to the correct size.
            #x_padded = tf.pad(self.x, paddings, "SYMMETRIC")
            # Encoder            in     num   shape  stride   pad
            enc  = conv2d(self.x,  32,  (5,5), (2,2),  "valid",
                           activation=relu, kernel_initializer=xavier())# 59x79
            print(f"enc1: {np.shape(enc)}")
            enc  = conv2d(enc, 62,  (5,5), (2,2),  "valid",
                           activation=relu, kernel_initializer=xavier())# 28x38
            print(f"enc2: {np.shape(enc)}")
            enc  = conv2d(enc, 128,  (4,4), (2,2),  "valid",
                           activation=relu, kernel_initializer=xavier())# 13x18
            print(f"enc3: {np.shape(enc)}")
            enc  = conv2d(enc, 256,  (4,4), (2,2),  "valid",
                           activation=relu, kernel_initializer=xavier())# 5x8
            print(f"enc4: {np.shape(enc)}")
            enc = flatten(enc)

        with tf.name_scope("sampling"):
            # VAE sampling
            '''
            Note: exp(log(log_sigma / 2)) = sigma
            '''
            self.mu           = dense(enc, self.embedding_dim, activation=None,
                                      kernel_initializer=xavier(),
                                      name="mu")
            self.log_sigma = dense(enc, self.embedding_dim, activation=None,
                                      kernel_initializer=xavier(),
                                      name="log_sigma")
            eps          = tf.random_normal(shape=tf.shape(self.mu),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32, name="eps")
            self.noisy_sigma  = tf.exp(self.log_sigma) * eps
            self.z       = tf.add(self.mu, self.noisy_sigma, name="z")

            #tf.summary.histogram("z", self.z)
            #tf.summary.histogram("log_sigma", self.log_sigma)
            #tf.summary.histogram("mu", self.mu)
            #tf.summary.histogram("eps", eps)

        with tf.name_scope("decoder"):
            # Decoder    in          num
            dec = dense(self.z, (256*5*7), activation=relu,
                             kernel_initializer=xavier())
            print(f"dec4: {np.shape(dec)}")
            dec  = tf.reshape(dec, (-1,5,7,256))
            print(f"dec3: {np.shape(dec)}")
            #                        in num  shape  stride   pad
            dec  = conv2d_transpose(dec, 128, (4,4), (2,2), "valid", activation=relu)
            print(f"dec2: {np.shape(dec)}")
            dec  = conv2d_transpose(dec,  64, (4,4), (2,2), "valid", activation=relu)
            print(f"dec1: {np.shape(dec)}")
            dec  = conv2d_transpose(dec,  32, (6,6), (2,2), "valid", activation=relu)
            print(f"dec0: {np.shape(dec)}")
            self.dec  = conv2d_transpose(dec, 3,  (10,18), (2,2), "valid",
                                         activation=sigmoid, name="reconstruction")
            print(f"out: {np.shape(self.dec)}")
#            self.dec  = relu(dec7, name="reconstruction")

        with tf.name_scope("loss"):
            # # VAE Loss
            # 1. Reconstruction loss: How far did we get from the actual image?
            #y_padded = tf.pad(self.y, paddings, "SYMMETRIC")
            self.rec_loss = tf.reduce_sum(tf.square(self.y - self.dec), axis=[1,2,3])
# Cross entropy loss from:                                          
            # https://stats.stackexchange.com/questions/332179/                 
            #         how-to-weight-kld-loss-vs-reconstruction-                 
            #         loss-in-variational-auto-encod                            
            # Must have self.dec activation as sigmoid                          
#            clipped = tf.clip_by_value(self.dec, 1e-8, 1-1e-8)                  
#            self.rec_loss = -tf.reduce_sum(self.y * tf.log(clipped)           
#                                      + (1-self.y) * tf.log(1-clipped), axis=[1,2,3])

            # 2. KL-Divergence: How far from the distribution 'z' is sampled
            #                   from the desired zero mean unit variance?
            self.kl_loss = 0.5 * tf.reduce_sum(tf.square(self.mu)
                                            + (tf.exp(2*self.log_sigma))
                                            - 2*self.log_sigma
                                            - 1
                                            , axis=1)
            self.loss =  tf.reduce_mean(self.rec_loss + self.kl_loss*self.beta,
                                        name='total_loss')

            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("kl_loss", tf.reduce_mean(self.kl_loss))
            tf.summary.scalar("rec_loss", tf.reduce_mean(self.rec_loss))
            tf.summary.scalar("beta", self.beta)

        update_tensor_dict(INPUTS,  IMAGE_INPUT,    self.x)
        update_tensor_dict(OUTPUTS, EMBEDDING,      self.z)
        update_tensor_dict(OUTPUTS, RECONSTRUCTION, self.dec)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_step = optimizer.minimize(self.loss, name="train_step")

        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def setup_meta(self, save_dir, test_gen):
        logs_path = os.path.join(save_dir, "train_logs")
        metadata  = os.path.join(logs_path, "metadata.tsv")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        test_gen.reset(shuffle=False)
        t_test = trange(test_gen.steps_per_epoch)
        t_test.set_description('tf.projector meta')
        labels = []
        for _ in t_test:
            batch  = test_gen.get_next_batch()
            labels.extend([a["steering"] for a in batch["annotations"]])

        with open(metadata, 'w') as meta_file:
            for steering_bin in labels[:self.num_projections]:
                _bin = int(steering_bin)
                meta_file.write("{}\n".format(int(_bin)))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        embedding.metadata_path = metadata
        return logs_path


    def train(self, train_gen, test_gen, save_dir, epochs=10,
              sample_inf_gen=None, annealing_epochs=0, beta_max=1):
        return_info = {"graph_path"    : "",
                        "ckpt_path"    : "",
                        "out_path"     : save_dir,
                        "tensor_json"  : ""}
        #logs_path = self.setup_meta(save_dir, test_gen)
        logs_path = os.path.join(save_dir, "train_logs")

        # Tensorboard
        merge = tf.summary.merge_all()

        best_ckpt = ""
        with tf.Session(config=tf.ConfigProto(gpu_options=GPU_OPTIONS)) as sess:
            # Tensorboard
            train_writer = tf.summary.FileWriter(logs_path+"/train_log", sess.graph)
            test_writer  = tf.summary.FileWriter(logs_path+"/test_log")

            # Init
            sess.run(self.init_vars)

            # some big number
            best_loss = 10**9.0

            # TODO: restart from checkpoint

            global_step = 0.
            for e in range(epochs):

                # Begin Training
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                sess.run(self.set_training, feed_dict={self._training: True})
                train_gen.reset()
                t_train = trange(train_gen.steps_per_epoch)
                #t_train = range(train_gen.steps_per_epoch)
                print(f"Training epoch: {e+1}")
                for step in t_train:
                    _beta = self.anneal_beta(global_step, annealing_epochs,
                                            train_gen.steps_per_epoch, beta_max)
                    sess.run(self.update_beta, feed_dict={self._beta: _beta})
                    ims, nims, _, _ = prepare_data(train_gen)
                    summary, _, loss, b = sess.run([merge, self.train_step,
                                                 self.loss, self.beta],
                               feed_dict={self.x: nims,
                                          self.y: ims})
                    train_writer.add_summary(summary, global_step)
                    t_train.set_description(f"{np.mean(loss):.3f}")
                    global_step += 1

                # Begin Testing
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                sess.run(self.set_training, feed_dict={self._training: False})
                embeddings      = []
                test_gen.reset(shuffle=True)
                #t_test = trange(test_gen.steps_per_epoch)
                #t_test.set_description('Testing')
                #t_test = range(test_gen.steps_per_epoch)
                #loss = []
                #for _ in t_test:
                ims, nims, _, _ = prepare_data(test_gen)
                summary, loss, zeds = sess.run([merge, self.loss, self.z],
                           feed_dict={self.x: nims,
                                      self.y: ims})
                #embeddings.extend(zeds)
                test_writer.add_summary(summary, global_step)
                print(f"Testing loss: {np.mean(loss):.3f}")
                #loss.append(_loss)

                # House keeping
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Update embedding being plotted by tensorboard
                #sess.run(self.update_embeddings,
                #         feed_dict={self._embeddings:
                #                    embeddings[:self.num_projections]})

                # Only begin saving checkpoints after beta is fully annealed
                if e >= annealing_epochs:
                    # only need to save graph once, then save the weights
                    # at each improving epoch
                    if e == annealing_epochs:
                        path = os.path.join(save_dir, "graph.meta")
                        self.saver.export_meta_graph(path)
                        return_info["graph_path"] = os.path.abspath(path)

                        path = write_tensor_dict_to_json(save_dir, tensor_dict)
                        return_info["tensor_json"] = os.path.abspath(path)

                    # If we have a new best loss then save the ckpt and some
                    # sample inferences
                    cur_loss = np.mean(loss)
                    print(f"Test Loss: {cur_loss:0.3f}\n" + 50*"-")
                    if cur_loss < best_loss:
                            best_loss = cur_loss
                            path = f"{save_dir}/ep_{e+1}.ckpt"
                            best_ckpt = self.saver.save(sess, path,
                                                        write_meta_graph=False)
                            return_info["ckpt_path"]=os.path.abspath(best_ckpt)
                            print("Model saved at {}".format(best_ckpt))
                            if (sample_inf_gen is not None):
                                path = os.path.join(*[save_dir,
                                               "sample_inferences", f"ep_{e}"])
                                self.save_sample_inference(sess,
                                                          sample_inf_gen, path)

        print(f"Done, final best loss: {best_loss:0.3f}")
        return_info["final_loss"] = float(best_loss)
        json.dump(return_info, open(save_dir+"/return_info.json", 'w'))
        return return_info


    def anneal_beta(self, step, annealing_epochs, steps_per_epoch, beta_max):
        if annealing_epochs==0: return beta_max
        return min(1.0, step/(annealing_epochs*steps_per_epoch)) * beta_max

    def save_sample_inference(self, sess, sample_inf_gen, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sample_inf_gen.reset(shuffle=False)
        batch    = sample_inf_gen.get_next_batch()
        names = batch["names"]
        nims = batch["noisy_images"] if "noisy_images" in batch else batch["images"]
        reconstructions = sess.run(self.dec,
                                   feed_dict={self.x: nims})
         
        save_images(reconstructions, names, save_dir)

# Static methods        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prepare_data(generator):
    batch    = generator.get_next_batch()
    # ToDo:
    # Add some noice to images maybe???
    images   = batch["images"]
    steering = [ele["steering"] for ele in batch["annotations"]]
    throttle = [ele["throttle"] for ele in batch["annotations"]]
    noisy_images = None
    if "noisy_images" in batch:
        noisy_images = batch["noisy_images"]
    else:
        noisy_images = images
    return images, noisy_images, steering, throttle

# def forward_pass(graph, input_tensor_name, output_tensor_names, vector_batch):
#     with tf.Session(graph=graph) as sess:
#         return sess.run(output_tensor_names, 
#                         feed_dict={input_tensor_name: vector_batch})
