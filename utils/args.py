import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--name', '-N', type=str, required=True)

    # Data
    parser.add_argument('--data_path', type=str, default="../NYUv2/")
    parser.add_argument('--dataset', type=str, choices=['NYUv2', 'NYU100', 'DSLR', 'SC', 'mobileDFD'], default='NYUv2')
    parser.add_argument('--normalize_dpt', action='store_true', default=False)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--image_num', type=int, default=5)
    parser.add_argument('--visible_image_num', type=int, default=5)
    parser.add_argument('--recon_all', action='store_true', default=False)
    parser.add_argument('--RGBFD', type=bool, default=True)
    parser.add_argument('--DPT', type=bool, default=True)
    parser.add_argument('--AIF', type=bool, default=False)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--fd', nargs='*', type=float, default=None)

    # Train
    parser.add_argument('--gt_dpt', action='store_true', default=False)
    parser.add_argument('--gt_aif', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--BS', '-B', type=int, default=8)
    parser.add_argument('--epochs', '-E', type=int, default=2000)
    parser.add_argument('-W', type=int, default=10)
    parser.add_argument('-D', type=int, default=3)
    parser.add_argument('-input_channels', type=int, default=5)
    parser.add_argument('-output_channels', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default='./model/funet/')

    parser.add_argument('--blur_loss_lambda', type=float, default=0)
    parser.add_argument('--recon_loss_lambda', type=float, default=1)
    parser.add_argument('--sm_loss_lambda', type=float, default=1e1)
    parser.add_argument('--sharp_loss_lambda', type=float, default=0)
    parser.add_argument('--recon_loss_alpha', type=float, default=0.15)
    parser.add_argument('--cstack_loss_beta', type=float, default=0.1)
    parser.add_argument('--sm_loss_beta', type=float, default=1.)
    parser.add_argument('--blur_loss_sigma', type=float, default=1.)
    parser.add_argument('--blur_loss_window', type=int, default=7)

    parser.add_argument('--aif_recon_loss_alpha', type=float, default=0.85)
    parser.add_argument('--aif_recon_loss_lambda', type=float, default=1e1)
    parser.add_argument('--aif_blur_loss_lambda', type=float, default=1)

    parser.add_argument('--dpt_post_op', type=str, choices=['raw', 'clip', 'norm'], default='norm')
    parser.add_argument('--continue_from', type=str, default=None)
    # parser.add_argument('--eval_from_load', type=str, default='/mnt/petrelfs/fapinzhe/CEGAN/model/funet/2022-08-19-11-50-29_fs/model_0.157524.bin')
    parser.add_argument('--eval_from_load', type=str, default='/mnt/petrelfs/fapinzhe/CEGAN/model/funet/2022-08-23-01-21-20/model_0.296206.bin')


    parser.add_argument('--eval', action='store_true', default=False)

    # Render
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--soft_merge', action='store_true', default=False)

    # Camera
    parser.add_argument('--fnumber', type=float, default=1)
    parser.add_argument('--focal_length', type=float, default=2.9e-2)
    parser.add_argument('--sensor_size', type=float, default=3.1*1e-2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--camera_near', type=float, default=0.1)
    parser.add_argument('--camera_far', type=float, default=10.)

    # Logging
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--VIS_FREQ', type=int, default=100)

    # Saving
    parser.add_argument('--SAVE_FREQ', type=int, default=500)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
    parser.add_argument('--img_save_dir', type=str, default='./image/')

    # Misc
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--early_stop', type=int, default=None)
    parser.add_argument('--manual_seed', type=int, default=None)

    return parser
