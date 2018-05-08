from vae import Model
import os
from utils import *
from generator import DataGenerator, preprocess_normalize_images_bin_annos
from generator import prepare_batch_images_and_labels
import json

'''
Exaple bash command:

python train.py --train-txt
'''
def save_config(save_dir, data_dir, train_txt, test_txt, lr, batch_size,
                epochs, in_shape, best_ckpt, best_loss, beta, annealing_epochs,
                message):
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
    payload["best_loss"]   = float(best_loss)
    payload["annealing"]   = annealing_epochs
    payload["beta"]        = beta
    payload["message"]     = message
    path = os.path.join(save_dir, "config.json")

    with open(path, 'w') as f:
        json.dump(payload, f)

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-txt', type=str, required=True,
                        help="Path to list of training example names.")
    parser.add_argument('--test-txt', type=str, required=True,
                        help="Path to list of test example names.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="Directory you wish to save training resuts to.")
    parser.add_argument('--lr', type=float, required=False, default = 0.001,
                        help="Learning rate.")
    parser.add_argument('--batch-size', type=int, required=False, default=50,
                        help="Number of samples fed into the network at one time.")
    parser.add_argument('--epochs', type=int, required=False, default=10,
                       help="Number of times the network looks at all the data.")
    parser.add_argument('--beta', type=int, required=False, default=1,
                       help="Weighting applied to KL divergence loss")
    parser.add_argument('--annealing', type=int, required=False, default=0,
                       help="Number epochs with which to increase beta from 0 \
                             to beta.")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--shape', type=int, required=False, nargs=3,
                        default = [120, 160, 3],
                        help="height width chanels")
    parser.add_argument('--message', type=str, required=True,
                       help="an reminder or other data you may need to \
                             identify the training run later.")
    args        = parser.parse_args()
    data_dir    = args.data_dir
    image_dir   = os.path.join(data_dir, "images")
    anno_dir    = os.path.join(data_dir, "annotations")
    train_path  = args.train_txt
    test_path   = args.test_txt
    batch_size  = args.batch_size

    # Load list of image names for train and test
    raw_train       = load_dataset(train_path)
    raw_test        = load_dataset(test_path)


    # Create train and test generators
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

    sample_gen  = DataGenerator(batch_size=10,
                      data_set=raw_test[:10],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      preprocess_fn=preprocess_normalize_images_bin_annos,
                      prepare_batch_fn=prepare_batch_images_and_labels,
                      shuffle=False)

    # Save the config to a file
    epochs      = args.epochs
    in_shape    = args.shape
    lr          = args.lr
    save_dir    = args.save_dir
    message     = args.message
    best_loss   = -1
    annealing_epochs = args.annealing
    beta        = args.beta
    best_ckpt   = "The session must have crashed before finnishing :-("

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_config(save_dir, data_dir, train_path, test_path, lr, batch_size,
                epochs, in_shape, best_ckpt, best_loss, beta, annealing_epochs,
                message)

    # Kick-off
    vae         = Model(in_shape)
    ckpt_loss   = vae.train(train_gen, test_gen, save_dir,
                            epochs=epochs, lr=lr, sample_inf_gen=sample_gen,
                            annealing_epochs=annealing_epochs, beta_max=beta)
    best_ckpt = ckpt_loss["best_ckpt"]
    best_loss = ckpt_loss["best_loss"]
    save_config(save_dir, data_dir, train_path, test_path, lr, batch_size,
                epochs, in_shape, best_ckpt, best_loss, beta, annealing_epochs,
                message)

if __name__ == "__main__":
    main()
