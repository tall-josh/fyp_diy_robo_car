from vae import Model
import os
from utils import *
from vae_data_generator import DataGenerator
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

def save_config(save_dir, data_dir, train_txt, test_txt, lr, batch_size, epochs, in_shape, best_ckpt, message):
    payload = {}
#    payload["name"]        = name
    payload["data_dir"]    = data_dir
    payload["train_txt"]   = train_txt
    payload["test_txt"]    = test_txt
    payload["lr"]          = lr
    payload["batch_size"]  = batch_size
    payload["epochs"]      = epochs
    payload["in_shape"]    = in_shape
    payload["best_ckpt"]   = best_ckpt
    payload["message"]     = message
    path = os.path.join(save_dir, "config.json")

    with open(path, 'w') as f:
        json.dump(payload, f)

''' -----       VAE       -----

python train.py --train-txt ../data//evened_train.txt --test-txt ../data/evened_test.txt --save-dir vae_000 --epochs 1 --data-dir ../data/clr_120_160/images/ --shape 120 160 3 --message "test run, totally will not work"
'''

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-txt', type=str, required=True,
                        help="Path to list of training example names.")
    parser.add_argument('--test-txt', type=str, required=True,
                        help="Path to list of test example names.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="Directory you wish to save training results to. This must be unique and will be created for you at run time. ")
    parser.add_argument('--lr', type=float, required=False, default = 0.001,
                        help="Learning rate.")
    parser.add_argument('--batch-size', type=int, required=False, default=50,
                        help="Number of samples fed into the network at one time.")
    parser.add_argument('--epochs', type=int, required=False, default=10,
                       help="Number of times the network looks at all the data.")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--shape', type=int, required=False, nargs=3, default = [120, 160, 3],
                        help="height width chanels")
    parser.add_argument('--message', type=str, required=True,
                       help="an reminder or other data you may need to identify the training run later.")
    NUM_BINS    = 15
    args        = parser.parse_args()
    data_dir    = args.data_dir
    image_dir   = os.path.join(data_dir, "images")
    anno_dir    = os.path.join(data_dir, "annotations")
    train_path  = args.train_txt
    test_path   = args.test_txt

    # Load list of image names for train and test
    train       = load_dataset(train_path)
    test        = load_dataset(test_path)


    # Create train and test generators
    batch_size  = args.batch_size
    train_gen   = DataGenerator(batch_size=batch_size,
                      data_set=train[:2000],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      num_bins=NUM_BINS)

    test_gen    = DataGenerator(batch_size=batch_size,
                      data_set=test[:500],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      num_bins=NUM_BINS)

    sample_gen  = DataGenerator(batch_size=10,
                      data_set=test[:10],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      num_bins=NUM_BINS,
                      shuffle=False)

    # Save the config to a file
    epochs      = args.epochs
    in_shape    = args.shape
    lr          = args.lr
    save_dir    = args.save_dir
    message     = args.message
    best_ckpt   = "The session must have crashed before finnishing :-("
    assert_message = "Name must be unique, This will be the name of the dir we'll used to save checkpoints"
    assert not os.path.exists(save_dir), "{}: {}".format(assert_message, save_dir)
    os.makedirs(save_dir)
    save_config(save_dir, data_dir, train_path, test_path, lr, batch_size, epochs, in_shape, best_ckpt,  message)

    # Kick-off
    vae         = Model(in_shape)
    best_ckpt   = vae.Train(train_gen, test_gen, save_dir,
                            epochs=epochs, lr=lr, sample_inf_gen=sample_gen)
    save_config(save_dir, data_dir, train_path, test_path, lr, batch_size, epochs, in_shape, best_ckpt,  message)

if __name__ == "__main__":
    main()
