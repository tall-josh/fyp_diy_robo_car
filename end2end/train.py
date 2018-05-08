from tf_donkey import Model
import os
from utils import *
from generator import DataGenerator, preprocess_normalize_images_bin_annos
from generator import prepare_batch_images_and_labels
import json

'''
{
"train_text"  : "../data/evened_train.txt",
"trest_text"  : "../data/evened_test.txt",
"name"        : "cnn_test",
"save_dir"    : "donkey_car",
"num_bins"    : 15,
"data_dir"    : "..data/color_120_160",
"in_shape"    : [120, 160, 3],
"message"     : "a simple nn test",
"epochs"      : 1,
"lr"          : 0.001
}
'''

def save_config(save_dir, data_dir, num_bins, lr, batch_size, epochs, in_shape, best_ckpt, message):
    payload = {}
#    payload["name"]        = name
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
    parser.add_argument('--num-bins', type=int, required=True)
    parser.add_argument('--lr', type=float, required=False, default = 0.001)
    parser.add_argument('--batch-size', type=int, required=False, default=50)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--shape', type=int, required=True, nargs=3, help="height width chanels")
    parser.add_argument('--message', type=str, required=True)

    args        = parser.parse_args()
    data_dir    = args.data_dir
    image_dir   = os.path.join(data_dir, "images/")
    anno_dir    = os.path.join(data_dir, "annotations/")
    train_path  = args.train_txt
    test_path   = args.test_txt

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
                      preprocess_fn=preprocess_normalize_images_bin_annos,
                      prepare_batch_fn=prepare_batch_images_and_labels)
    test_gen    = DataGenerator(batch_size=batch_size,
                      data_set=raw_test[:50],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      preprocess_fn=preprocess_normalize_images_bin_annos,
                      prepare_batch_fn=prepare_batch_images_and_labels)

    # Kick-off
    #name        = args.name
    save_dir    = args.save_dir
    epochs      = args.epochs
    in_shape    = args.shape
    lr          = args.lr
    classes     = [i for i in range(num_bins)]
    message     = args.message

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_ckpt   = "must have crashed during training :-("
    save_config(save_dir, data_dir, num_bins, lr, batch_size, epochs,
                in_shape, best_ckpt,  message)
    car_brain   = Model(in_shape, classes=classes)
    best_ckpt   = car_brain.train(train_gen, test_gen, save_dir, epochs=epochs)


if __name__ == "__main__":
    main()
