from tf_donkey import Model

def VideoDrive(ckpt, video_path, num_bins=15, in_shape=(120,160,3),
               car_ip="192.168.43.56", port=5555, pwm_min_max=(-0.5, 0.5), steering_range_deg=40):

    classes = [x for x in range(num_bins)]
    car_brain = Model("test_name", in_shape=in_shape, classes=classes)
    car_brain.VideoDrive(ckpt, video_path, car_ip)

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--num-bins', type=int, required=True)
    parser.add_argument('--vid-w', type=int,    required=False, default=160)
    parser.add_argument('--vid-h', type=int,    required=False, default=120)
    parser.add_argument('--vid-chan', type=int, required=False, default=3)
    parser.add_argument('--car-ip', type=str, required=True)
    parser.add_argument('--port', type=float, required=True)
    parser.add_argument('--steering-range-deg', type=float, required=False, default=40)
    parser.add_argument('--pwm-min', type=float, required=False, default=-0.5)
    parser.add_argument('--pwm-max', type=float, required=False, default=0.5)

    args        = parser.parse_args()
    mode        = args.mode
    ckpt        = args.ckpt_path
    #ckpt        = "ep_19-step_161-loss_0.944.ckpt"
    video_path  = args.video_path
    #video_path  = "/home/jp/Documents/FYP/ml/data/videoplayback.mp4"
    num_bins    = args.num_bins
    in_shape    = [args.vid_h, args.vid_w, args.vid_chan]
    car_ip      = args.car_ip
    port        = args.port
    pwm_min_max = (args.pwm_min, args.pwm_max)
    steering_range_deg = args.steering_range_deg

    print("mode: {}".format(args.mode))
    if mode == "video-drive":
#        print(f"ckpt:               {ckpt}")
#        print(f"video_path:         {video_path}")
#        print(f"num_bins:           {num_bins}")
#        print(f"in_shape:           {in_shape}")
#        print(f"car_ip:             {car_ip}")
#        print(f"port:               {port}")
#        print(f"pwm_min_max:        {pwm_min_max}")
#        print(f"steering_range_deg: {steering_range_deg}")
        VideoDrive(ckpt, video_path, num_bins, in_shape, car_ip, port, pwm_min_max, steering_range_deg)
    else:
        print(f"mode: '{mode}' is not valid :-(")

if __name__ == "__main__":
    main()
