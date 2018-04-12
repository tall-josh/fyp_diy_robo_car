import os
import glob
import numpy as np
import matplotlib.image as matimg
import random
import json
import io
import csv
from tqdm import trange, tqdm

'''load a single image'''
def load_image(path):
    img = matimg.imread(path)
    return img

'''load a single annotation'''
def load_anno(path):
    with open(path, 'r') as f:
        anno = json.load(f)
    return anno

'''get a collection of paths from a directory'''
def get_paths_glob(path_to_dir):
    paths = glob.glob(os.path.join(path_to_dir))
    try:
        assert(len(paths) > 0)
    except AssertionError as e:
        e.args += (" :-( ERROR!!! --> {} is not a path or the dir \
                 is empty.".format(path_to_dir),)
        raise
    return paths

def load_sample(image_dir, anno_dir, name):
    raise ValueError("Deprecated, now call 'load_image_anno_pair()' ")

'''Load all images and annotations from list of paths'''
def load_image_anno_pair(image_dir, anno_dir, name):
    im_path = os.path.join(image_dir, f"{name}.jpg")
    an_path = os.path.join(anno_dir, f"{name}.json")
    return load_image(im_path), load_anno(an_path)
    
'''load just images from list of paths'''
def load_images(im_paths):
    ims  = []
    for im in im_paths:
        ims.append(load_image(im))
    return ims

'''extract the name of an image given the path'''
def get_data_point_name(data_point_path):
    # some/path/to/image.jpg
    sp = data_point_path.split('/')
    # [some path to image.jpg]
    sp = sp[-1].split('.')
    # [image jpg]
    return sp[0]

'''
    Loads image and annotatios from data_set_path
'''
def load_dataset(data_set_path):
    data = []
    with open(data_set_path, 'r') as data_set:
        data_reader = csv.reader(data_set, delimiter=',')
        for name in data_reader:
            data.append(name[0])
    return data
    
    
''' 
- Bin the steering angles.
  bin_number = data_value // bin_size
  ie: 0 to 14 with 5 bins is 3 elements per bin
'''
def bin_steering_anno(steering_anno, num_bins, val_range=1024):
    bin_size = val_range / num_bins
    return steering_anno // bin_size
    #ANNO_IDX = 1
    #result = []
    #bin_size = val_range/num_bins
    #for i,a in data_set:
    #    result.append((i, a // bin_size))
    #return result
    
    # Magic numbers to convert RGB to gray scale
def rgb2gray(data):
    gray = []
    for rgb in data:
        temp = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        temp = np.expand_dims(temp, axis=2)
        gray.append(temp)
    gray = np.asarray(gray)
    return gray

def get_data_point_names_sequnetial(annos_dir, count=None):
    '''
    Loads image and annotation names and checks that all images have an annotation.
    Shuffles names. If count is not None then only count elements will be returned.
    '''
    
    an_paths = get_paths_glob(os.path.join(annos_dir, "*.json")) 
    
    if count is not None:
        an_paths = an_paths[:count]
    
    an_names = [get_data_point_name(i) for i in an_paths]
    result = []

    #don't worry about this. It just allows me to show a loading bar
    print("Importing images and annotations.")
    an_names.sort()
    
    pbar = tqdm(list(range(len(an_names)-1)))  
    for name, _ in zip(an_names, pbar):
        anno = load_anno(os.path.join(annos_dir,name+".json"))
        result.append(anno)
            
    return result


'''def check_input_image_dimentions_equal(dataset, known_shape=None):
    shape = known_shape if known_shape is not None else np.shape(dataset[0])

    for img in dataset:
        try:
            next_img_shape = np.shape(img)
            assert(shape == next_img_shape)
        except AssertionError as e:
            e.args += (" :-( ERROR --> All input images MUST be of the same size. We have detected at least 1 descrepency. size: {} vs size: {}".format(shape, np.shape(img)),)
            raise
        shape = next_img_shape
    return shape'''


