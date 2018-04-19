import numpy as np
import csv
import io
from utils import *
from tqdm import trange

class BaseDataGenerator():
    
    def __init__(self, batch_size, data_set, image_dir, anno_dir=None, shuffle=True):
        self.data_set        = data_set
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir if anno_dir is not None else image_dir
        self._indexes        = list(range(len(data_set)))
        self.steps_per_epoch = len(self.data_set) // self.batch_size
        self.data            = None 
        
        self.reset(shuffle)
        
    def reset(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self._indexes)
        else:
            self._indexes = list(range(len(self.data_set)))
        self.current_step    = 0
    
    def normalize_image(self, image):
        image = (image / 255.0)
        return image
    
    def augment(self, image, anno):
        raise NoImplementationError()
    
    def get_next_batch(self):
        raise NoImplementationError()
        
    def load_all_data(self, image_path, anno_path):
        raise NoImplementationError()
    

class NoImplementationError(Exception):
    def __init__(self, method_name):
        super().__init__(f"Method is abstract, so requires an implementation.")