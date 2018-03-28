import cv2
import argparse
import os
import numpy as np


def extract_binary_encoding(image, rois, h=3, w=4):
    binary_string = ""
    thresh = 20
    for i, roi in enumerate(rois):
        x0, y0 = roi[0],     roi[1]
        x1, y1 = roi[0] + w, roi[1] + h
        cell = cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        binary_string += "0" if np.mean(cell) < thresh else "1"
        # cv2.imwrite("{}/im{}_cell{}.jpg".format(out_path, count, i), cell)
        # cv2.rectangle(image, (x0,y0), (x1, y1), (0,0,255), thickness=1)
    number = int(binary_string, 2)
    # print("binary: {} = {}".format(binary_string, number))
    return number, binary_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True,
                        help='Path to video')
    parser.add_argument('--out-w', type=float, required=False, default=None,
                        help='output width image. If None, then original width is used.')
    parser.add_argument('--out-h', type=float, required=False, default=None,
                        help='output width image. If None, then original height is used.')
    parser.add_argument('--out-path', type=str, required=True,
                        help='Where to save the resulting output.')
    #parser.add_argument('--to-gray', type=bool, required=False, default=True)
  # a boolean option
    parser.add_argument("--to-gray",
        action="store_true",
        default=False
    )
    args    = parser.parse_args()
    path    = args.in_path
    out_path = args.out_path
    to_gray = args.to_gray
    try:
        assert os.path.exists(out_path), " :-( --- {} is not a path you drongo!".format(out_path)
    except AssertionError as e:
        print(e)

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    if success:
        while success:

            rois = [(115,2),(124,2),(133,2),(141,2),(149,2),
                    (158,2),(166,2),(174,2),(182,2),(191,2)]
            enc_0, bin_0 = extract_binary_encoding(image, rois)
            # print("binary: {} = {}".format(binary_string, number))

            rois = [(237,2), (246,2), (254,2), (262,2), (270,2),
                    (279,2), (287,2), (295,2), (304,2), (313,2)]
            enc_1, bin_1 = extract_binary_encoding(image, rois)
            # print("binary: {} = {}".format(binary_string, number))
            if to_gray:
                image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            img_cropped = image[40:,:]
            orig_h      = img_cropped.shape[:2][0]
            orig_w      = img_cropped.shape[:2][1]
            out_h       = args.out_h if args.out_h is not None else orig_h
            out_w       = args.out_w if args.out_w is not None else orig_w
            x_scale     = out_w / orig_w
            y_scale     = out_h / orig_h
            img_resized = cv2.resize(img_cropped, None, fx=x_scale, fy=y_scale)
            cv2.imwrite("{}/images/frame{:05}.jpg".format(out_path, count), img_resized)
            with open("{}/annotations/frame{:05}.txt".format(out_path, count),
            'a') as f:
                f.write("frame{:05}, {}, {}\n".format(count, enc_0, enc_1))

            count += 1
            success,image = vidcap.read()
            # if count == 5:
            #     success = False
            if count%100 == 0:
                print("Processed {} frames.".format(count))

        else:
            print("DANM Dauuug, looks like the file was not read properly.")
            print("The path you input was:\n{}".format(path))
            
if __name__ == "__main__":
    main()
