import numpy as np
import csv
import io
from utils import *
from base_data_generator import BaseDataGenerator
from tqdm import tqdm

'''
Generator for producing batches of image, annotation pairs for training
a classifyer style networks.
'''
class DataGenerator(BaseDataGenerator):
    
    def __init__(self, batch_size, data_set, image_dir, shuffle=True, count=None):
        
        super().__init__(batch_size, data_set, image_dir, 
                         shuffle=shuffle)
        
        self.data = self.load_all_data(image_dir, data_set)
                
    
    def augment(self, image, anno):
        pass
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        # if random.uniform(0.,1.) < 0.5:
        #     return np.flip(image, 1)
        # return image_aug, image
    
    def load_all_data(self, image_dir, data_set):
        all_data = []
        pbar = tqdm(data_set)
        pbar.set_description("Loading Data")
        for name in pbar:
            im = load_image(os.path.join(image_dir, name+".jpg"))
            im = self.normalize_image(np.array(im))
            pair = {}
            pair["original_image"]     = im
            pair["augmented_image"]    = im
            pair["name"]               = name
            # To Do, add noisy image or something
            # pari["augmented_image"] = self.augment(im)
            all_data.append(pair)
        return all_data
    
    def get_next_batch(self):
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None
        
        i = self.current_step * self.batch_size
        images = []
        annos  = []
        names  = []
        string = ""
        for ele in range(self.batch_size):
            pair     = self.data[self._indexes[i+ele]]
            image    = pair["original_image"]
            anno     = pair["augmented_image"]
            name     = pair["name"]
            
            images.append(image)
            annos.append(anno)
            names.append(name)
        
        self.current_step += 1
        
        # if gray scale add single channel dim
        if len(np.shape(images)) == 3:
            images = np.expand_dims(images, axis=3)
            images = np.expand_dims(annos, axis=3)
            
        return images, annos, names

