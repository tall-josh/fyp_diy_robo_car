import sys
sys.path.append('..')
from modular_network import ClassifierModule
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
import os
from utils import *
from behavoural_data_generator import DataGenerator
import json

def save_config(save_dir, data_dir, num_bins, lr, batch_size, epochs, in_shape, best_ckpt, message):
    payload = {}
    payload["data_dir"]    = data_dir
    payload["num_bins"]    = num_bins
    payload["lr"]          = lr
    payload["batch_size"]  = batch_size
    payload["epochs"]      = epochs
    payload["in_shape"]    = in_shape
    payload["best_ckpt"]   = best_ckpt
    payload["message"]     = message
    path = os.path.join(save_dir, "config.json")

    with open(path, 'w') as f:
        json.dump(payload, f)

'''-----      Behavoural      -----
Poop
'''
def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-txt', type=str, required=True)
    parser.add_argument('--test-txt', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--num-bins', type=int, required=False, default=15)
    parser.add_argument('--lr', type=float, required=False, default = 0.001)
    parser.add_argument('--batch-size', type=int, required=False, default=50)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--message', type=str, required=True)

    args        = parser.parse_args()
    data_dir    = args.data_dir
    image_dir   = os.path.join(data_dir, "images/")
    anno_dir    = os.path.join(data_dir, "annotations/")
    train_path  = args.train_txt
    test_path   = args.test_txt
    encoder = {"path"               : args.encoder,
               "input_tensor_name"  : "x:0",
               "output_tensor_name" : "sampling/z:0",
               "name"               : "vae"}

    # Load list of image names for train and test
    raw_train   = load_dataset(train_path)
    raw_test    = load_dataset(test_path)


    # Create train and test generators
    num_bins    = args.num_bins
    batch_size  = args.batch_size
    train_gen   = DataGenerator(batch_size=batch_size,
                      data_set=raw_train[:200],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      num_bins=num_bins)

    test_gen    = DataGenerator(batch_size=batch_size,
                      data_set=raw_test[:100],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      num_bins=num_bins)

    # Kick-off
    #name        = args.name
    save_dir    = args.save_dir
    epochs      = args.epochs
    in_shape    = [120,160,3]
    lr          = args.lr
    classes     = [i for i in range(num_bins)]
    message     = args.message

    assert_message = "Name must be unique, This will be the name of the dir we'll used to save checkpoints"
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    best_ckpt   = "must have crashed during training :-("
    save_config(save_dir, data_dir, num_bins, lr, batch_size, epochs, in_shape, best_ckpt,  message)
    layer_def   = []
    layer_def.append({"neurons"     : 30,
                       "activation" : tf.nn.relu,
                       "name"       : "mod1",
                       "init"       : xavier(),
                       "dropout"    : 1.})
    layer_def.append({"neurons"     : 15,
                       "activation" : None,
                       "name"       : "logits",
                       "init"       : xavier(),
                       "dropout"    : 1.})

    car_brain   = ClassifierModule(encoder, layer_def, classes=classes)
    best_ckpt   = car_brain.train(train_gen, test_gen, save_dir, epochs)


if __name__ == "__main__":
    main()
