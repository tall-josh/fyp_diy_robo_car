import cv2
import argparse
import os
import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Takes an --in-dir containing \
    a collection of images and applies a resize oporation, then\
    saves in --out-dir.')
    parser.add_argument('--in-dir', type=str,
                        help='dir containing a sequence of images.')
    parser.add_argument('--out-w', type=float, default=1,
                        help='output width image')
    parser.add_argument('--out-h', type=float, default=1,
                        help='output height image')
    parser.add_argument('--out-dir', type=str,
                        help='Where to save your results.')
    args = parser.parse_args()

    base_dir = args.in_dir
    out_w    = float(args.out_w)
    out_h    = float(args.out_h)
    out_dir  = args.out_dir


def resize_image():
im_paths = os.walk(base_dir)
im_paths = list(next(im_paths))
im_paths = im_paths[2]

# out / in = scale
minus_encoding = 40 # number of pixels to crop from top to remove encoding
count = 0
for im in im_paths:
    try:
        orig   = cv2.imread(os.path.join(*[base_dir, im]))
        orig_h = orig.shape[:2][0]
        orig_w = orig.shape[:2][1]
        orig= orig[minus_encoding:, :]
        x_scale = out_w / orig_w
        y_scale = out_h / (orig_h-minus_encoding)

        resized = cv2.resize(orig, None, fx=x_scale, fy=y_scale)
        cv2.imwrite(os.path.join(*[out_dir, "resized_{}".format(im)]), resized)
        count += 1
        # print("{} -> {}".format(orig.shape[:2],resized.shape[:2]))
        # print("x{} -> y{}".format(orig.shape[:2][0],orig.shape[:2][1]))
        if count % 20 == 0:
            print("Resized {} or {} images...w,h:{}".format(count, len(im_paths), (resized.shape[:2][1],resized.shape[:2][0]) ))
    except AttributeError as e:
        print("Error, something went wrong with image ({})...skipping: message: {}".format(im, e))
