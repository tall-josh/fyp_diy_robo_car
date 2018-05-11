from vae import Model
import os
from utils import *
import json
from freeze_graph import freeze_meta
#from generator import DataGenerator
from generator import DenoisingVaeGenerator as DataGenerator

# For training (WILL bin steering annos, and WILL normalize throttle)           
# Images are normalized                                                         
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
from generator import preprocess_normalize_images_bin_annos as process_fn       
from generator import prepare_batch_images_and_labels_RAND_MIRROR as prep_batch 

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
    payload["best_loss"]   = best_loss
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
    parser.add_argument('--emb-dim', type=int, required=True)
    parser.add_argument('--beta', type=float, required=False, default=1,
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
    noisy_dir   = os.path.join(data_dir, "noisy")
    anno_dir    = os.path.join(data_dir, "annotations")
    train_path  = args.train_txt
    test_path   = args.test_txt
    batch_size  = args.batch_size

    # Load list of image names for train and test
    raw_train       = load_dataset(train_path)
    raw_test        = load_dataset(test_path)


    # Create train and test generators
    train_gen   = DataGenerator(batch_size=batch_size,
                      data_set=raw_train,
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      noisy_dir=noisy_dir,
                      preprocess_fn=process_fn,
                      prepare_batch_fn=prep_batch)

    test_gen    = DataGenerator(batch_size=500,
                      data_set=raw_test[:500],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      noisy_dir=noisy_dir,
                      preprocess_fn=process_fn,
                      prepare_batch_fn=prep_batch)

    sample_gen  = DataGenerator(batch_size=20,
                      data_set=raw_test[:20],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      noisy_dir=noisy_dir,
                      preprocess_fn=process_fn,
                      prepare_batch_fn=prep_batch)
    sample_gen.reset(shuffle=False)

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
    vae         = Model(in_shape, embedding_dim=args.emb_dim)
    #(self, in_shape, lr=0.001, embedding_dim=10, num_projections=20):
    return_info = vae.train(train_gen, test_gen, save_dir,
                                epochs=epochs, sample_inf_gen=sample_gen,
                                annealing_epochs=annealing_epochs, beta_max=beta)
    frozen_meta = freeze_meta(return_info["graph_path"],
                                    return_info["ckpt_path"],
                                    return_info["out_path"]+"/frozen.pb",
                                    return_info["tensor_json"])
    return_info["frozen_meta"] = frozen_meta

    best_ckpt = return_info["final_loss"]
    best_loss = return_info["ckpt_path"]
    save_config(save_dir, data_dir, train_path, test_path, lr, batch_size,
                epochs, in_shape, best_ckpt, best_loss, beta, annealing_epochs,
                message)
    json.dump(return_info, open(save_dir+"/return_info.json", 'w'))

if __name__ == "__main__":
    main()
