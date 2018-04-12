import numpy as np
import csv
import io
from utils import *
from base_data_generator import BaseDataGenerator

'''
Generator for producing batches of image pairs for training
a VAE or Autoencoder style network. 
'''
class DataGenerator(BaseDataGenerator):
    
    def __init__(self, batch_size, data_set, image_dir, 
                 anno_dir=None, shuffle=True):
        
        '''Generally this will occure, I have left the  "anno_dir"
        argument here in case someone for some reason want to store
        noisy samply images for training on disk seperate to the
        desired reconstruction images.'''
        if anno_dir is None:
            anno_dir = image_dir
        
        super().__init__(batch_size, data_set, image_dir, 
                         anno_dir, shuffle=shuffle)
        self.image_dir       = image_dir
    
    def augment(self, image, anno):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        if random.uniform(0.,1.) < 0.5:
            return np.flip(image, 1), np.flip(anno, 1)
        return image, anno
        
    def get_next_batch(self):
        #print("step {} of {}".format(self.current_step, self.steps_per_epoch))
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None
        
        i = self.current_step * self.batch_size
        images = []
        annos  = [] 
        for ele in range(self.batch_size):
            name = self.data_set[i+ele]
            
            image = load_image(os.path.join(self.image_dir, f"{name}.jpg"))
            anno  = load_image(os.path.join(self.anno_dir, f"{name}.jpg"))
            
            if self.shuffle:
                image, anno = self.augment(image, anno)
                
            images.append(image)
            annos.append(anno)
        
        self.current_step += 1
        
        # if gray scale add single channel dim
        if len(np.shape(images)) == 3:
            images = np.expand_dims(images, axis=3)
            annos  = np.expand_dims(annos , axis=3)
            
        return images, annos

