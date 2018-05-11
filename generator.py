import numpy as np
import csv
import io
from utils import *
from tqdm import tqdm

NUM_BINS = 15

def preprocess_normalize_images_bin_annos(im, an):
    num_bins  = NUM_BINS
    im = normalize_image(np.array(im))
    an["steering"] = bin_value(an["steering"],
                               num_bins,
                               val_range=1024)
    an["throttle"] = an["throttle"] / 1024
    pair = {}
    pair["original_image"]  = im
    pair["anno"]            = an
    return pair

def preprocess_normalize_images_only(im, an):
    im = normalize_image(np.array(im))
    pair = {}
    pair["original_image"]  = im
    pair["anno"]            = an
    return pair

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
def prepare_batch_vae(gen_ref, count):
    batch = {"images"  : [],
             "augmented_images" : [],
             "annotations"      : [],
             "names"            : []}
    string = ""
    for ele in range(gen_ref.batch_size):
        pair = gen_ref.data[gen_ref._indexes[count+ele]]
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
'''
def prepare_batch_images_and_labels_NO_MIRROR(gen_ref, marker):
    num_bins = NUM_BINS
    batch = {'images'      : [],
             'annotations' : [],
             'names'       : []}
    for ele in range(gen_ref.batch_size):
        pair     = gen_ref.data[gen_ref._indexes[marker+ele]]
        im, an = pair["original_image"], pair["anno"]
        batch["images"].append(im)
        batch["annotations"].append(an)
        batch["names"].append(pair["name"])

 # if gray scale add single channel dim
    if len(np.shape(batch["images"])) == 3:
        batch["images"] = np.expand_dims(batch["images"], axis=3)
    return batch

def prepare_batch_images_and_labels_RAND_MIRROR(gen_ref, marker):
    num_bins = NUM_BINS
    batch = {'images'      : [],
             'annotations' : [],
             'names'       : []}
    for ele in range(gen_ref.batch_size):
        pair     = gen_ref.data[gen_ref._indexes[marker+ele]]
        im, an = pair["original_image"], pair["anno"]
        im, an = mirror_at_random(im, an, num_bins)
        batch["images"].append(im)
        batch["annotations"].append(an)
        batch["names"].append(pair["name"])

 # if gray scale add single channel dim
    if len(np.shape(batch["images"])) == 3:
        batch["images"] = np.expand_dims(batch["images"], axis=3)
    return batch

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mirror_at_random(image, anno, num_bins):
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

def normalize_image(image, zero_mean=False):
        if zero_mean:
            image = (image / 255.0) - 0.5
        else:
            image = (image / 255.0)
        return image
'''
def load_all_data(image_dir, anno_dir, data_set):
        all_data = []
        pbar = tqdm(data_set)
        pbar.set_description("Loading Data")
        for name in pbar:
            im, an = load_image_anno_pair(image_dir, anno_dir, name)
            pair = {}
            pair["original_image"]  = im
            pair["anno"]            = an
            pair["name"]            = name
            # To Do, add noisy image or something
            # pari["augmented_image"] = self.augment(im)
            all_data.append(pair)
        return all_data
'''
'''
Generator for producing batches of image, annotation pairs for training
classifyer style networks.
'''
class DataGenerator(object):

    def __init__(self, batch_size, data_set, image_dir, anno_dir, shuffle=True,
                 preprocess_fn=None, prepare_batch_fn=None):

        self.preprocess      = preprocess_fn
        self.prepare_batch   = prepare_batch_fn
        self.data_set        = data_set
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir if anno_dir is not None else image_dir
        self._indexes        = list(range(len(data_set)))
        self.steps_per_epoch = len(self.data_set) // self.batch_size
        self.data = self.load_all_data(image_dir, anno_dir, data_set)
        self.reset(shuffle)

    def reset(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self._indexes)
        else:
            self._indexes = list(range(len(self.data_set)))
        self.current_step    = 0

    def load_all_data(self, image_dir, anno_dir, data_set):
        all_data = []
        pbar = tqdm(data_set)
        pbar.set_description("Loading Data")
        for name in pbar:
            im, an = load_image_anno_pair(image_dir, anno_dir, name)
            pair = {}
            if self.preprocess is not None:
                pair = self.preprocess(im, an)
            else:
                pair['image'] = im
                pair['anno']  = an
            pair["name"] =  name
            all_data.append(pair.copy())
        return all_data

    def get_next_batch(self, augment=True):
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None

        marker = self.current_step * self.batch_size
        batch = self.prepare_batch(self, marker)
        self.current_step += 1
        return batch

class DenoisingVaeGenerator(object):

    def __init__(self, batch_size, data_set, image_dir, anno_dir, noisy_dir=None, shuffle=True,
                 preprocess_fn=None, prepare_batch_fn=None):

        #self.preprocess      = preprocess_fn
        #self.prepare_batch   = prepare_batch_fn
        self.data_set        = data_set
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir if anno_dir is not None else image_dir
        self._indexes        = list(range(len(data_set)))
        self.steps_per_epoch = len(self.data_set) // self.batch_size
        self.data = self.load_all_data(image_dir, noisy_dir, anno_dir, data_set)
        self.reset(shuffle)

    def reset(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self._indexes)
        else:
            self._indexes = list(range(len(self.data_set)))
        self.current_step    = 0

    def preprocess_normalize_images(self, im, an, noisy_im):
        im = normalize_image(np.array(im))
        noisy_im = normalize_image(np.array(noisy_im))
        pair = {}
        pair["original_image"]  = im
        pair["anno"]            = an
        pair["noisy_image"]     = noisy_im
        return pair

    def load_all_data(self, image_dir, noisy_dir, anno_dir, data_set):
        all_data = []
        pbar = tqdm(data_set)
        pbar.set_description("Loading Data")
        for name in pbar:
            im, an = load_image_anno_pair(image_dir, anno_dir, name)
            n_im   = load_image(os.path.join(noisy_dir,name+".jpg")) 
            pair = {}
            pair = self.preprocess_normalize_images(im, an, n_im)
            pair["name"] =  name
            all_data.append(pair.copy())
        return all_data

    def mirror_at_random(self, image, noisy_image, anno, num_bins):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        # Create copy so not to modify the original data
        anno_aug = anno.copy()
        steering_original = anno_aug["steering"]
        if random.uniform(0.,1.) < 0.5:
            image     = np.flip(image, 1)
            noisy_image     = np.flip(noisy_image, 1)
            anno_aug["steering"] = 1023-steering_original
        return image, noisy_image, anno_aug

    def add_100s_and_1000s(self, noisy_images):
        noise = np.random.choice([1, 0], size=np.prod(np.shape(noisy_images)), p=[0.7, 0.3])
        noise = np.reshape(noise, np.shape(noisy_image))
        noisy_images *= noise
        return noisy_images

    def get_next_batch(self, augment=True):
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None

        marker = self.current_step * self.batch_size
        num_bins = NUM_BINS
        batch = {'images'      : [],
                 'noisy_images' : [],
                 'annotations' : [],
                 'names'       : []}
        for ele in range(self.batch_size):
            pair     = self.data[self._indexes[marker+ele]]
            im, nim, an = pair["original_image"], pair["noisy_image"], pair["anno"]
            im, nim, an = self.mirror_at_random(im, nim, an, num_bins)
            batch["images"].append(im)
            batch["noisy_images"].append(nim)
            batch["annotations"].append(an)
            batch["names"].append(pair["name"])

        batch["noisy_images"] = add_100s_and_1000s(batch["noisy_images"])
        # if gray scale add single channel dim
        if len(np.shape(batch["images"])) == 3:
            batch["images"] = np.expand_dims(batch["images"], axis=3)
            batch["noisy_images"] = np.expand_dims(batch["noisy_images"], axis=3)
        self.current_step += 1
        return batch

