import numpy as np
import csv
import io
from utils import *
from base_data_generator import BaseDataGenerator

'''
Generator for producing batches of image, annotation pairs for training
a classifyer style networks.
'''
class DataGenerator(BaseDataGenerator):
    
    def __init__(self, batch_size, data_set, image_dir, 
                 anno_dir, num_bins, shuffle=True):
        
        super().__init__(batch_size, data_set, image_dir, 
                         anno_dir, shuffle=shuffle)
        self.num_bins = num_bins
    

    def augment(self, image, anno):
        '''
        Mirror randomly, add some noise, pick up milk.
        '''
        if random.uniform(0.,1.) < 0.5:
            return (np.flip(image, 1), (self.num_bins-1) - anno)
        return image, anno
    
    def all_annotations(self, sort=False):
        annos  = {"steering": [], "throttle": []}
        for name in self.data_set:
            path = os.path.join(self.anno_dir, f"{name}.json")
            anno = load_anno(path)
            annos['steering'].append(anno['steering'])
            annos['throttle'].append(anno['throttle'])
        return annos
    
    def get_next_batch(self):
        #print("step {} of {}".format(self.current_step, self.steps_per_epoch))
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None
        
        i = self.current_step * self.batch_size
        images = []
        annos  = {"steering": [], "throttle": []}
        for ele in range(self.batch_size):
            name = self.data_set[i+ele]
            image, anno = load_sample(self.image_dir, self.anno_dir, name)
            steering = bin_steering_anno(anno['steering'], self.num_bins, val_range=1024)
            throttle = anno['throttle']
            if self.shuffle:
                image, steering = self.augment(image, steering)
                
            images.append(image)
            annos['steering'].append(steering)
            annos['throttle'].append(throttle)
        
        self.current_step += 1
        
        # if gray scale add single channel dim
        if len(np.shape(images)) == 3:
            images = np.expand_dims(images, axis=3)
            
        return images, annos

