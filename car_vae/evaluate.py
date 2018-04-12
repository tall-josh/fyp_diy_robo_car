from tf_donkey import Model
import os
from utils import *
from data_generator import DataGenerator
import json

def load_config(path):
    with open(path, 'r') as f:
        payload = json.load(f)
    return payload

#    payload["name"]        = name
#    payload["data_dir"]    = base_dir
#    payload["num_bins"]    = num_bins
#    payload["lr"]          = lr
#    payload["batch_size"]  = batch_size
#    payload["epochs"]      = epochs
#    payload["in_shape"]    = in_shape
#    payload["best_ckpt"]   = best_ckpt
#    payload["message"]     = message

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-txt', type=str, required=True)
    parser.add_argument('--config-dir', type=str, required=True)
    args = parser.parse_args()
    config = load_config(os.path.join(args.config_dir, "config.json"))
    print(config)
    data_dir    = config["data_dir"]
    image_dir   = os.path.join(data_dir, "images/")
    anno_dir    = os.path.join(data_dir, "annotations/")
    eval_path   = args.eval_txt
    num_bins    = config["num_bins"]
    # Load list of image names for train and test
    raw_eval    = load_dataset(eval_path)
    
    # Create train and test generators
    batch_size  = config['batch_size']
    eval_gen    = DataGenerator(batch_size=batch_size, 
                      data_set=raw_eval,
                      image_dir=image_dir,
                      anno_dir=anno_dir, 
                      num_bins=num_bins)
    
    # Kick-off
    save_dir    = args.config_dir
    in_shape    = config['in_shape']
    ckpt        = config['best_ckpt']
    classes     = [i for i in range(num_bins)]
    car_brain   = Model(in_shape, classes=classes)
    evaluation  = car_brain.Evaluate(eval_gen, ckpt, save_figs=True, save_dir=os.path.join(save_dir, "eval"))
    

if __name__ == "__main__":
    main()
