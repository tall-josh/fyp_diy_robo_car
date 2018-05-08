from concrete_modules import ThrottleModule as Module
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
import os
from utils import *
import json

from freeze_graph import freeze_meta
from generator import DataGenerator as gen
# For training (WILL bin steering annos, and WILL normalize throttle)           
# Images are normalized                                                         
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
from generator import preprocess_normalize_images_bin_annos as process_fn       
from generator import prepare_batch_images_and_labels_RAND_MIRROR as prep_batch 
                                                                                
# For evaluation (will NOT bin steering annos, and will leave throttle 0-1024)  
# Images are normalized                                                         
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
#from generator import preprocess_normalize_images_only as process_fn           
#from generator import prepare_batch_images_and_labels_NO_MIRROR as prep_batch


'''
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
'''
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

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-txt', type=str, required=True)
    parser.add_argument('--test-txt', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, required=False, default = 0.001)
    parser.add_argument('--batch-size', type=int, required=False, default=50)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--enc-pb', type=str, required=True)
    parser.add_argument('--tensor-json', type=str, required=True)
    parser.add_argument('--message', type=str, required=True)

    args        = parser.parse_args()
    data_dir    = args.data_dir
    image_dir   = os.path.join(data_dir, "images/")
    anno_dir    = os.path.join(data_dir, "annotations/")
    train_path  = args.train_txt
    test_path   = args.test_txt
    
    froz_enc_pb = args.enc_pb
    tensor_names = json.load(open(args.tensor_json, 'r'))
    encoder = {"path"               : froz_enc_pb,
               "input_tensor_name"  : tensor_names['inputs']['image_input'],
               "output_tensor_name" : tensor_names['outputs']['embedding'],
               "name"               : "vae"}

    # Load list of image names for train and test
    raw_train   = load_dataset(train_path)
    raw_test    = load_dataset(test_path)


    # Create train and test generators
    batch_size  = args.batch_size
    
    train_gen=gen(batch_size=10, 
              data_set=raw_train[:100],
              image_dir=image_dir,
              anno_dir=anno_dir,
              preprocess_fn=process_fn,
              prepare_batch_fn=prep_batch)

    test_gen=gen(batch_size=10, 
                 data_set=raw_test[:50],
                 image_dir=image_dir,
                 anno_dir=anno_dir,
                 preprocess_fn=process_fn,
                 prepare_batch_fn=prep_batch)

    # Kick-off
    NUM_BINS = 15
    #name        = args.name
    save_dir    = args.save_dir
    epochs      = args.epochs
    in_shape    = [120,160,3]
    lr          = args.lr
    classes     = [i for i in range(NUM_BINS)]
    message     = args.message

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    best_ckpt   = "must have crashed during training :-("
    save_config(save_dir, data_dir, NUM_BINS, lr, batch_size, epochs, in_shape, best_ckpt,  message)
    layer_def   = []
    layer_def.append({"neurons"     : 30,
                       "activation" : tf.nn.relu,
                       "name"       : "mod1",
                       "init"       : xavier(),
                       "dropout"    : 1.})
    layer_def.append({"neurons"     : 1,
                       "activation" : None,
                       "name"       : "logits",
                       "init"       : xavier(),
                       "dropout"    : 1.})

    car_brain   = Module(encoder, layer_def, classes=classes)
    return_info = car_brain.train(train_gen, test_gen, save_dir, epochs)
    frozen_meta = freeze_meta(return_info["graph_path"],
                            return_info["ckpt_path"],
                            return_info["out_path"]+"/frozen.pb",
                            return_info["tensor_json"])


if __name__ == "__main__":
    main()
