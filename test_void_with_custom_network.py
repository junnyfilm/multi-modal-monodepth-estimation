# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Shift window testing and flip testing is modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from configs.config import Config
import utils.logging as logging
import utils.metrics as metrics
from models.model import IDEDepth
from dataset.void_dataset_v2 import void_dataset_v2

from utils.viz_utils import Visualize_CV
from utils.util import normalize_rot_vector


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'pose_mse_r', 'pose_mse_t']


def main_custom():
    args = Config()
    args.settings_for_test_void_with_custom_network()

    args.ckpt_dir = os.path.join(args.result_dir, 'train/ckpt/checkpoint_best.pth')

    args.save_visualize = True
    args.do_evaluate = True

    if args.save_visualize:
        result_path = os.path.join(args.result_dir, 'test')
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

    if args.do_evaluate:
        result_metrics = {}
        for metric in metric_name:
            result_metrics[metric] = 0.0

    print("\n1. Define Model")
    args.model_scale = 32
    model = IDEDepth(args=args).cuda()

    model_weight = torch.load(args.ckpt_dir)
    model.load_state_dict(model_weight['model_state_dict'])
    model.eval()

    print("\n2. Define Dataloader")
    test_dataset = void_dataset_v2(cfg=args, data_path=args.customdata, crop_size=(args.crop_h, args.crop_w), is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    viz_tools = Visualize_CV()

    print("\n3. Inference & Evaluate")
    for batch_idx, batch in enumerate(test_loader):
        frame1 = batch['frame1'].cuda()
        frame2 = batch['frame2'].cuda()
        depth_gt1 = batch['depth_image1'].cuda()
        depth_gt2 = batch['depth_image2'].cuda()
        raw_image1 = batch['raw_image1'][0]
        raw_image2 = batch['raw_image2'][0]
        rel_pose = batch['rel_pose'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():
            preds = model(frame1, frame2)

        pred_d1 = preds['pred_d1'].squeeze()
        depth_gt1 = depth_gt1.squeeze()

        # Make the determinant of rotation matrix 1
        preds['out_p'][..., :9] = normalize_rot_vector(preds['out_p'][..., :9])
        rel_pose[..., :9] = normalize_rot_vector(rel_pose[..., :9])

        if args.do_evaluate:
            pred_crop, gt_crop = metrics.cropping_img(args, pred_d1, depth_gt1)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            computed_mse = metrics.eval_pose(preds['out_p'], rel_pose)

            for key in computed_result.keys():
                result_metrics[key] += computed_result[key]
            for key in computed_mse.keys():
                result_metrics[key] += computed_mse[key]

        if args.save_visualize:
            place, scene = filename.split('/')[-2:]
            save_dir = os.path.join(result_path, 'depth', place)
            os.makedirs(save_dir, exist_ok=True)

            pred_d_numpy = pred_d1.squeeze().cpu().numpy()
            pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)
            pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)

            depth_gt1_numpy = depth_gt1.squeeze().cpu().numpy()
            depth_gt1_numpy = (depth_gt1_numpy / depth_gt1_numpy.max()) * 255
            depth_gt1_numpy = depth_gt1_numpy.astype(np.uint8)
            depth_gt1color = cv2.applyColorMap(depth_gt1_numpy, cv2.COLORMAP_RAINBOW)

            error_map_numpy = np.abs(depth_gt1_numpy - pred_d_numpy)
            error_map_color = cv2.applyColorMap(error_map_numpy, cv2.COLORMAP_WINTER)
            viz_tools.save_results(raw_image1, raw_image2, depth_gt1color, pred_d_color, error_map_color, save_dir, scene)

            pos_save_dir = os.path.join(result_path, 'pose', place)
            viz_tools.save_pos_results(rel_pose, preds['out_p'], pos_save_dir, scene)

    if args.do_evaluate:
        for key in result_metrics.keys():
            result_metrics[key] = result_metrics[key] / (batch_idx + 1)
        display_result = logging.display_result(result_metrics)
        print(display_result)

    print("Done")


if __name__ == "__main__":
    main_custom()
