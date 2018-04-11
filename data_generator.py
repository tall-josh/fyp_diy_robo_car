import numpy as np
import csv
import io
from utils import *

class DataGenerator():
    
    def __init__(self, batch_size, data_set, image_dir, anno_dir, num_bins, shuffle=True, debug=None):
        self.__unsized_data_set = data_set
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir
        self.steps_per_epoch = len(self.__unsized_data_set) // self.batch_size
        self.current_step    = 0
        self.num_bins        = num_bins
        self.shuffle         = shuffle
        self.debug           = debug
        self.data_set = self.size_dataset_to_suit_batch_size()
    
    def size_dataset_to_suit_batch_size(self):
        
        if self.shuffle:
            np.random.shuffle(self.__unsized_data_set)
        else: 
            self.__unsized_data_set.sort()
        
        to_remove = len(self.__unsized_data_set) % self.batch_size
        if to_remove != 0:
#            print("Warning! number of training samples ({}) is not divisable \
#                   by batch_size ({})".format(len(self.__unsized_data_set), self.batch_size))
#            print("Removing {} samples from data set.".format(to_remove))
#            print("UPDATED size is: {}".format(len(self.__unsized_data_set) - to_remove))
            return self.__unsized_data_set[:-to_remove]
        else:
            return self.__unsized_data_set[:]
    
    def reset(self, shuffle=True):
        self.shuffle=shuffle
        self.data_set = self.size_dataset_to_suit_batch_size()
        self.current_step    = 0
    
    def still_has_data(self):
        return self.current_step < self.steps_per_epoch
    
    #def get_steps_per_epoch(self):
    #    return self.steps_per_epoch
    
    def norm_data(images):
        result = images.astype('float32') / 255.
        return result

    def mirror_at_random(self, image, anno):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        if random.uniform(0.,1.) < 0.5:
            return (np.flip(image, 1), (self.num_bins-1) - anno)
        return image, anno
    
    def _all_annotations(self, sort=False):
        annos = []
        for _, anno in self.data_set:
            annos.append(anno)
        return annos
    
    def get_next_batch(self):
        #print("step {} of {}".format(self.current_step, self.steps_per_epoch))
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None
        
        i = self.current_step * self.batch_size
        images = []
        annos  = []
        for ele in range(self.batch_size):
            im_name, anno = self.data_set[i+ele]
            image = load_image(self.image_dir + im_name + ".jpg")
# DEBUGGING random mirror
#             zeros = np.zeros((80,40))
#             ones  = np.ones((80,40))
#             image = np.concatenate((zeros, ones), axis=1)
#             anno  = 7
            if self.shuffle:
                image, anno = self.mirror_at_random(image, anno)
            images.append(image)
            annos.append(anno)
        
        self.current_step += 1
        
        # if gray scale add single channel dim
        if len(np.shape(images)) == 3:
            images = np.expand_dims(images, axis=3)
            
        return images, annos

