import cv2
import argparse
import os
import numpy as np


def split_video_and_process_frames(path, out_path, to_gray=False):

    try:
        assert os.path.exists(out_path), " :-( --- {} is not a path you drongo!".format(out_path)
    except AssertionError as e:
        print(e)

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    if success:
        while success:

            if to_gray:
                image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            cv2.imwrite("{}/frame{:06}.jpg".format(out_path, count), image)
            count += 1
            success,image = vidcap.read()
            if count%1000 == 0:
                print("Processed {} frames.".format(count))
        print("You Little Rippa!!!!!!!")
    else:
        print("DANM Dauuug, looks like the file was not read properly.")
        print("The path you input was:\n{}".format(path))
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True,
                        help='Path to video')
    parser.add_argument('--out-path', type=str, required=True,
                        help='Where to save the resulting output.')
    parser.add_argument("--to-gray",
        action="store_true",
        default=False
    )
    args    = parser.parse_args()
    in_path    = args.in_path
    out_path = args.out_path
    to_gray = args.to_gray
    split_video_and_process_frames(in_path, out_path, to_gray)
            
if __name__ == "__main__":
    main()
