import numpy as np
import csv
import io
from utils import *

class BaseDataGenerator():
    
    def __init__(self, batch_size, data_set, image_dir, anno_dir, shuffle=True):
        self.__unsized_data_set = data_set
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir
        self.steps_per_epoch = len(self.__unsized_data_set) // self.batch_size
        self.current_step    = 0
        self.shuffle         = shuffle
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
    
#    def norm_image(self, images):
#        result = images.astype('float32') / 255.
#        return result

    def augment(self, image, anno):
        raise NoImplementationError()
    
    def get_next_batch(self):
        raise NoImplementationError()

class NoImplementationError(Exception):
    def __init__(self, method_name):
        super().__init__(f"Method is abstract, so requires an implementation.")