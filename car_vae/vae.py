import cv2
import tensorflow as tf
from tensorflow.python.layers.core import dense, dropout, flatten
from tensorflow.python.layers.convolutional import conv2d, conv2d_transpose
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import trange, tqdm
import numpy as np
import os
import sys
sys.path.append('..')
#from show_graph import show_graph
from metrics import Metrics
from utils import save_images

NUM_EMBEDDINGS = 500

class Model:
    def __init__(self, in_shape):
        '''
        classes:  List of class names or integers corrisponding to each class being classified
                  by the network. ie: ['left', 'straight', 'right'] or [0, 1, 2]
        '''
        # Define model
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="input")
        self.y = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="label")
        self.training   = tf.placeholder(tf.bool, name="training")
        self.new_embeddings = tf.placeholder(tf.float32, shape=[None, 50])
        self.embeddings     = tf.Variable(np.zeros(shape=[NUM_EMBEDDINGS,50]), name="embeddings", dtype=tf.float32)
        paddings = tf.constant([[0,0],[4,4],[0,0],[0,0]])
        relu    = tf.nn.relu
        sigmoid = tf.nn.sigmoid

        with tf.name_scope("encoder"):
            # Padding invector so reconstruction returns to the correct size.
            x_padded = tf.pad(self.x, paddings, "SYMMETRIC")
            y_padded = tf.pad(self.y, paddings, "SYMMETRIC")
            # Encoder            in     num   shape  stride   pad
            enc1  = conv2d(x_padded,  24,  (5,5), (2,2),  "same",
                              activation=relu, kernel_initializer=xavier(),
                              name="enc1")# 64, 80,24
            enc2  = conv2d(enc1, 32,  (5,5), (2,2),  "same",
                              activation=relu, kernel_initializer=xavier(),
                              name="enc2")# 32, 40,32
            enc3  = conv2d(enc2, 64,  (5,5), (2,2),  "same",
                              activation=relu, kernel_initializer=xavier(),
                              name="enc3")# 16, 20,64
            enc4  = conv2d(enc3, 64,  (3,3), (2,2),  "same",
                              activation=relu, kernel_initializer=xavier(),
                              name="enc4")#  8, 10,64
            enc5  = conv2d(enc4, 64,  (3,3), (1,1),  "same",
                              activation=relu, kernel_initializer=xavier(),
                              name="enc5")#  8, 10,64
            enc5f = flatten(enc5)
            #                     in   num
            enc6  = dense(enc5f, 100,
                             activation=relu, kernel_initializer=xavier(),
                             name="enc6")
            enc6d = dropout(enc6, rate=0.1, training=self.training)

        with tf.name_scope("sampling"):
            # VAE sampling
            gamma        = 1.0
            self.mu           = dense(enc6d, 50, activation=None,
                                      kernel_initializer=xavier(),
                                      name="mu")
            self.log_sigma_sq = dense(enc6d, 50, activation=None,
                                      kernel_initializer=xavier(),
                                      name="log_sigma_sq")
            eps          = tf.random_normal(shape=tf.shape(self.mu),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32)
            # Sample A
            #self.noisy_sigma  = tf.sqrt(tf.exp(self.log_sigma_sq))*eps
            #_z       = tf.add(self.mu, self.noisy_sigma, name="z")

            # Sample B
            self.z = self.mu + tf.exp(self.log_sigma_sq /2) * eps
#           self.z = dropout(_z, rate=0.1, training=self.training)

            tf.summary.histogram("z", self.z)
            tf.summary.histogram("log_sigma_sq", self.log_sigma_sq)
            tf.summary.histogram("mu", self.mu)
            tf.summary.histogram("eps", eps)

        with tf.name_scope("decoder"):
            # Decoder    in          num
            dec1 = dense(self.z, 100, activation=relu,
                             kernel_initializer=xavier(), name="dec6")
            dec2 = dense(dec1, (8*10*64), activation=relu,
                             kernel_initializer=xavier(), name="dec7")
            dec2r  = tf.reshape(dec2, (-1,8,10,64))
            #                        in num  shape  stride   pad
            dec3  = conv2d_transpose(dec2r, 64, (3,3), (1,1), "same",
                                         activation=relu, name="dec5")
            dec4  = conv2d_transpose(dec3, 64, (3,3), (2,2), "same",
                                         activation=relu, name="dec4")
            dec5  = conv2d_transpose(dec4, 32, (5,5), (2,2), "same",
                                         activation=relu, name="dec3")
            dec6  = conv2d_transpose(dec5, 24, (5,5), (2,2), "same",
                                         activation=relu, name="dec2")
            self.dec  = conv2d_transpose(dec6, 3,  (5,5), (2,2), "same",
                                         activation=relu, name="dec1")

            # # VAE Loss
            # 1. Reconstruction loss: How far did we get from the actual image?
            self.rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_padded - self.dec), axis=1))

            # 2. KL-Divergence: How far from the "true" distribution of z's is
            #                   our parameterised z?
            # Loss A
            #self.kl_loss = tf.reduce_sum( 0.5 * (tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma) , axis=1)

            # Loss B
            print(np.shape(self.mu))
            print(np.shape(self.log_sigma_sq))
            _kl_loss = 1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq)
            self.kl_loss = -0.5 * tf.reduce_sum(_kl_loss, axis=1)


            beta = 0.05
            self.loss =  tf.reduce_mean(self.rec_loss + self.kl_loss)

            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("kl_loss", tf.reduce_mean(self.kl_loss))
            tf.summary.scalar("rec_loss", self.rec_loss)

            self.update_embeddings = self.embeddings.assign(self.new_embeddings)

        self.saver = tf.train.Saver()

    def Infer(self, images, checkpoint_path, save_dir):
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint_path)
            reconstructions = sess.run(self.dec, feed_dict={self.x: images, self.training: False})
            for i, (image, recon) in enumerate(zip(images,reconstructions)):
                pass
                # cv2.imwrite(os.path.join(save_dir, f"{i:001}_reconstruction.jpg"), recon)
                # cv2.imwrite(os.path.join(save_dir, f"{i:001}_original.jpg"), image)
            return reconstructions, images

    def _SetupMeta(self, save_dir, test_gen):
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
            for steering_bin in labels[:NUM_EMBEDDINGS]:
                _bin = int(steering_bin)
                meta_file.write("{}\n".format(int(_bin)))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        embedding.metadata_path = metadata
        return logs_path

    def Train(self, train_gen, test_gen, save_dir, epochs=10, lr=0.001, sample_inf_gen=None):

        logs_path = self._SetupMeta(save_dir, test_gen)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(self.loss)

        self.train_loss = {"total": [], "kl": [], "rec": []}
        self.test_loss  = {"total": [], "kl": [], "rec": []}
        best_ckpt = ""
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(logs_path, sess.graph)
            sess.run(tf.global_variables_initializer())
            best_loss = 10**9.0  # some big number
            # TODO: restart from checkpoint
            count = 0
            for e in range(epochs):
                merge = tf.summary.merge_all()
                temp_train_loss = []
                train_gen.reset()
                t_train = trange(train_gen.steps_per_epoch)
                t_train.set_description(f"Training Epoch: {e+1}")
                for step in t_train:
                    batch = train_gen.get_next_batch()
                    summary, _, loss, rec, kl = sess.run([merge, train_step, self.loss, self.rec_loss, self.kl_loss],
                               feed_dict={self.x: batch["augmented_images"],
                                          self.y: batch["original_images"],
                                          self.training: True})
                    temp_train_loss.append(np.array([loss, rec, np.mean(kl)]))
                    train_writer.add_summary(summary, count)
                    count += 1

                means = np.mean(np.array(temp_train_loss), axis=0)

                self.train_loss["total"].append(means[0])
                self.train_loss["rec"].append( means[1])
                self.train_loss["kl"].append(  means[2])
                cur_loss = means[0]
                print(f"Train Loss: {cur_loss}")

                test_gen.reset(shuffle=False)
                temp_test_loss = []
                embeddings      = []
                t_test = trange(test_gen.steps_per_epoch)
                t_test.set_description('Testing')
                for _ in t_test:
                    batch = test_gen.get_next_batch()
                    loss, rec, kl, zeds = sess.run([self.loss, self.rec_loss, self.kl_loss, self.z],
                               feed_dict={self.x: batch["augmented_images"],
                                          self.y: batch["original_images"],
                                          self.training: False})
                    temp_test_loss.append(np.array([loss, rec, np.mean(kl)]))
#                    print(f"mu: {mu[0]}")
#                    print(f"ns: {ns[0]}")
#                    print(f"zeds:  {zeds[0]}")
                    embeddings.extend(zeds)
                sess.run(self.update_embeddings, feed_dict={self.new_embeddings: embeddings[:NUM_EMBEDDINGS]})
                means = np.mean(np.array(temp_test_loss), axis=0)
                self.test_loss["total"].append(means[0])
                self.test_loss["rec"].append( means[1])
                self.test_loss["kl"].append(  means[2])

                cur_loss = means[0]
                print(f"Test Loss: {cur_loss:0.3f}")
                print("-"*50)
                if cur_loss < best_loss:
                        best_loss = cur_loss
                        path = f"{save_dir}/ep_{e+1}_loss_{best_loss:0.3f}.ckpt"
                        best_ckpt = self.saver.save(sess, path)
                        print("Model saved at {}".format(best_ckpt))

                        if (sample_inf_gen is not None):
                            path = os.path.join(*[save_dir, "sample_inferences", f"ep_{e}"])

                            self.SaveSampleInference(sess,sample_inf_gen, path)

        self.SaveLossPlots(save_dir)
        print(f"Done, final best loss: {best_loss:0.3f}")
        return best_ckpt

    def SaveSampleInference(self, sess, sample_inf_gen, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        batch = sample_inf_gen.get_next_batch()
        sample_inf_gen.reset(shuffle=False)
        reconstructions = sess.run(self.dec,
                                   feed_dict={self.x: batch["original_images"],
                                              self.training: False})
        names = [a["image"] for a in batch["annotations"]]
        save_images(reconstructions, names, save_dir)

    def SaveLossPlots(self, save_dir):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        import json

        for loss_type in self.train_loss.keys():
            # print(f"Train: {self.train_loss}")
            # print(f"Test:  {self.test_loss}")
            fig = plt.figure(figsize=(8,6))
            plt.plot(self.train_loss[loss_type], 'b-',  label="train")
            plt.plot(self.test_loss[loss_type],  'r--', label="test")
            plt.title(f'{loss_type} loss during training')
            plt.xlabel("Loss")
            plt.ylabel("Epoch")
            plt.legend()
            path = os.path.join(save_dir, f"{loss_type}_training_loss.jpg")
            fig.savefig(path)

            # Need to convet from numpy.float32 to native float32 for serialization
            self.train_loss[loss_type] = [float(x) for x in self.train_loss[loss_type]]
            self.test_loss[loss_type]  = [float(x) for x in self.test_loss[loss_type]]
        with open(os.path.join(save_dir, "train.json"), 'w') as f:
            json.dump(self.train_loss, f)
        with open(os.path.join(save_dir, "test.json"), 'w') as f:
            json.dump(self.test_loss, f)

"""
    def Evaluate(self, eval_gen, checkpoint_path, save_figs=False, save_dir=None):
        pass

    def TrainingResults(self):
        pass
        #return self.train_loss, self.test_loss, self.test_acc

    def Predict(self, images, checkpoint_path):
        pass
        #with tf.Session() as sess:
        #    self.saver.restore(sess,checkpoint_path)
        #    return sess.run([self.steering_probs, self.expected_bin], feed_dict={self.x:images, self.training: False})

    def GetGraph(self):
        return tf.get_default_graph().as_graph_def()

"""
