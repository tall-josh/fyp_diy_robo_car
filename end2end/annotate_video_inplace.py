'''
    - after running video_inference you'll need to 
      use something like ffmpeg to stitch the jpgs 
      into a video.
      
      try this from the dir at out_path:
      note the %05 for leading zeros. Make sure this is correct
      
      ffmpeg -i im_%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p ~/Desktop/stiched_video.mp4
'''
import cv2
import argparse
import os
import numpy as np


def ang_idx_to_rad(angle_idx, bin_range=128., max_steering_deg=45.):
    angle = angle_idx*128.       # convert back to 0-1023
    angle -= 1024./2.            # zero centre
    angle /= 1024./2             # normalize -1 to 1
    angle *= max_steering_deg    # to deg
    angle *= np.pi / 180.        # to rad
    return angle

def video_inference(sess, x, prediction, training, vid_path, out_path, nn_input_shape=(80,80,1)):
    net_in_w = nn_input_shape[0]
    net_in_h = nn_input_shape[1]
    try:
        assert os.path.exists(out_path), " :-( --- {} is not a path you drongo!".format(out_path)
    except AssertionError as e:
        print(e)

    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    print('Success: {}'.format(success))
    count = 0
    if success:
        
        while success:
            img_gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_cropped = img_gray[40:,:]
            orig_h      = img_cropped.shape[:2][0]
            orig_w      = img_cropped.shape[:2][1]
            #out_h       = args.out_h if args.out_h is not None else orig_h
            #out_w       = args.out_w if args.out_w is not None else orig_w
            x_scale     = net_in_w / orig_w
            y_scale     = net_in_h / orig_h
            img_resized = cv2.resize(img_cropped, None, fx=x_scale, fy=y_scale)
            img_resized = np.expand_dims(img_resized, axis=0)
            img_resized = np.expand_dims(img_resized, axis=3)
            ang_idx = sess.run(prediction,feed_dict={x: img_resized,  training: False})
            angle = ang_idx_to_rad(ang_idx, bin_range=128., max_steering_deg=50.)
            pt0 = (orig_w//2,orig_h//3)
            dx    =   60.*np.sin(angle)
            dy    =  -60.*np.cos(angle)
            pt1  = (int(pt0[0]+dx), int(pt0[1]+dy))
            
            #delta = (dx,dy)
            #pt1   = np.add(pt0, delta).astype('int32')
            #print(pt1)
            #print(dx)
            #print(dy)
            #print(pt0)
            #print(pt1)
            #print(ang_idx)
            image   = cv2.line(image, pt0,pt1,(0,0,255), 3)
            image   = cv2.putText(image, "{:.2}".format(ang_idx[0]), 
                                (240,65), cv2.FONT_HERSHEY_SIMPLEX,
                                1.,
                               (0,0,255),
                                2)
            #print(out_path)
            cv2.imwrite("{}/frame{:05}.jpg".format(out_path, count), image)
        
            count += 1
            success,image = vidcap.read()
            # if count == 5:
            #     success = False
            if count%100 == 0:
                print("Processed {} frames.".format(count))
    else:
        print("DANM Dauuug, looks like the file was not read properly.")
    
        print("The path you input was:\n{}".format(vid_path))

'''
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

    args    = parser.parse_args()
    path    = args.in_path
    out_path = args.out_path
    
if __name__ == "__main__":
    main()
'''