import cv2
import argparse
import os
import re
from sorting import *

parser = argparse.ArgumentParser(description='Takes an --in-dir containing \
a sequence of images. Subtracts one image from another spaced --stride appart\
and saves is --out-dir.')
parser.add_argument('--in-dir', type=str,
                    help='dir containing a sequence of images.')
parser.add_argument('--stride', type=int, default=1,
                    help='number of frames to skip')
parser.add_argument('--out-dir', type=str,
                    help='Where to save your results.')
args = parser.parse_args()

base_dir = args.in_dir
stride   = args.stride
out_dir  = args.out_dir

im_paths = os.walk(base_dir)
im_paths = list(next(im_paths))
im_paths = im_paths[2]

    
sort_nicely(im_paths)
count = 0
for im0, im1 in zip(im_paths[:-stride:stride], im_paths[stride::stride]):
  print("{} - {}".format(im1, im0))
  image0 = cv2.imread(os.path.join(*[base_dir, im0]))
  image1 = cv2.imread(os.path.join(*[base_dir, im1]))
  diff   = image0 - image1
  diff[diff<0] = 0
  print("{}, {}, {}".format(image1[100,100], image0[100,100], diff[100,100]))
  cv2.imwrite(os.path.join(*[out_dir, "diff{}.jpg".format(count)]), diff)
  count += 1
