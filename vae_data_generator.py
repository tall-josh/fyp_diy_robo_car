import numpy as np
import io
from utils import *
from base_data_generator import BaseDataGenerator
from tqdm import tqdm

'''
Generator for producing batches of image, annotation pairs for training
a classifyer style networks.
'''
class DataGenerator(BaseDataGenerator):

    def __init__(self, batch_size, data_set, image_dir, anno_dir, num_bins, shuffle=True, count=None):

        super().__init__(batch_size, data_set, image_dir, anno_dir,
                         shuffle=shuffle)
        self.data = self.load_all_data(image_dir, anno_dir, data_set, num_bins)

    def augment(self, image, anno, num_bins):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        if random.uniform(0.,1.) < 0.5:
            image     = np.flip(image, 1)
            augmented = image.copy()
            anno["steering"] = (num_bins-1)-anno["steering"]
        else:
            image     = np.flip(image, 1)
            augmented = image.copy()
#           TODO: AUGMENT
            anno["steering"] = (num_bins-1)-anno["steering"]
        return image, augmented, anno

    def load_all_data(self, image_dir, anno_dir, data_set, num_bins):
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
            im, aug_im, an          = self.augment(im, an, num_bins)
            pair["original_image"]  = im
            pair["augmented_image"] = aug_im
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
        batch = {"original_images"  : [],
                 "augmented_images" : [],
                 "annotations"      : [],
                 "names"            : []}
        string = ""
        for ele in range(self.batch_size):
            pair     = self.data[self._indexes[i+ele]]
            batch["original_images"].append(pair["original_image"])
            batch["augmented_images"].append(pair["augmented_image"])
            batch["annotations"].append(pair["anno"])
            batch["names"].append(pair["name"])
        self.current_step += 1

        # if gray scale add single channel dim
        if len(np.shape(batch["original_images"])) == 3:
            batch["original_images"] = np.expand_dims(batch["original_images"], axis=3)
            batch["augmented_images"] = np.expand_dims(batch["augmented_images"], axis=3)
        return batch

