import numpy as np
import csv
import io
from utils import *
from tqdm import tqdm

'''
Generator for producing batches of image, annotation pairs for training
classifyer style networks.
'''
class BaseDataGenerator(object):

    def __init__(self, batch_size, data_set, image_dir,
                 anno_dir, num_bins, shuffle=True, count=None):

        self.data_set        = data_set
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir if anno_dir is not None else image_dir
        self._indexes        = list(range(len(data_set)))
        self.steps_per_epoch = len(self.data_set) // self.batch_size
        self.num_bins        = num_bins
        self.data = self.load_all_data(image_dir, anno_dir, data_set,                                                      num_bins=num_bins)
        self.reset(shuffle)
        
    def normalize_image(self, image, zero_mean=False):
        if zero_mean:
            image = (image / 255.0) - 0.5
        else:
            image = (image / 255.0)
        return image
    
    def reset(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self._indexes)
        else:
            self._indexes = list(range(len(self.data_set)))
        self.current_step    = 0
    
    def augment(self, image, anno, num_bins):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        # Create copy so not to modify the original data
        anno_aug = anno.copy()
        bin_original = anno_aug["steering"]
        if random.uniform(0.,1.) < 0.5:
            image     = np.flip(image, 1)
            anno_aug["steering"] = (num_bins-1)-bin_original
#       TODO: AUGMENT
        return image, anno_aug

    def load_all_data(self, image_dir, anno_dir, data_set, num_bins):
        self.num_bins = num_bins
        all_data = []
        pbar = tqdm(data_set)
        pbar.set_description("Loading Data")
        for name in pbar:
            im, an = load_image_anno_pair(image_dir, anno_dir, name)
            im = self.normalize_image(np.array(im))
            pair = {}
            an["steering"]          = bin_value(an["steering"],
                                                num_bins,
                                                val_range=1024)
            pair["original_image"]  = im
            pair["anno"]            = an
            pair["name"]            = name
            # To Do, add noisy image or something
            # pari["augmented_image"] = self.augment(im)
            all_data.append(pair)
        return all_data

    def get_next_batch(self):
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None

        i = self.current_step * self.batch_size
        batch = {"images"  : [],
                 "annotations"      : [],
                 "names"            : []}
        string = ""
        for ele in range(self.batch_size):
            pair     = self.data[self._indexes[i+ele]]
            im, an   = self.augment(pair["original_image"],
                                          pair["anno"], self.num_bins)
            batch["images"].append(im)
            batch["annotations"].append(an)
            batch["names"].append(pair["name"])
        self.current_step += 1

        # if gray scale add single channel dim
        if len(np.shape(batch["images"])) == 3:
            batch["images"] = np.expand_dims(batch["images"], axis=3)
        return batch
    
###################################################################################

'''
Generator for producing batches of image, augmented_image, annotation sets for training autoencoder style networks.
'''
class VaeDataGenerator(BaseDataGenerator):

    def augment(self, image, anno, num_bins):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        # Create copy so not to modify the original data
        anno_aug = anno.copy()
        bin_original = anno_aug["steering"]
        if random.uniform(0.,1.) < 0.5:
            image     = np.flip(image, 1)
            anno_aug["steering"] = (num_bins-1)-bin_original
        
        # ToDo: Further augmentation
        image_aug = image
        return image, image_aug, anno_aug
    
    def get_next_batch(self):
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None

        i = self.current_step * self.batch_size
        batch = {"images"  : [],
                 "augmented_images" : [],
                 "annotations"      : [],
                 "names"            : []}
        string = ""
        for ele in range(self.batch_size):
            pair     = self.data[self._indexes[i+ele]]
            im, im_aug, anno = self.augment(pair["original_image"],
                                            pair["anno"],
                                            self.num_bins)
            
            batch["images"].append(im)
            batch["augmented_images"].append(im_aug)
            batch["annotations"].append(anno)
            batch["names"].append(pair["name"])
        self.current_step += 1

        # if gray scale add single channel dim
        if len(np.shape(batch["images"])) == 3:
            batch["images"] = np.expand_dims(batch["images"], axis=3)
            batch["augmented_images"] = np.expand_dims(batch["augmented_images"], axis=3)
        return batch