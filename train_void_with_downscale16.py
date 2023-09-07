# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import torch
import shutil

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import utils.metrics as metrics
import utils.logging as logging

from configs.config import Config
from models.model import IDEDepth
from models.optimizer import build_optimizers
from dataset.void_dataset_v2 import void_dataset_v2
from utils.criterion import SiLogLoss, WeightedMSELoss
from utils.util import save_model, save_model_best_rmse_model, normalize_rot_vector
from torch.nn.utils.rnn import pad_sequence

from utils.viz_utils import Visualize_CV

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'pose_mse_r', 'pose_mse_t']

def collate_fn_imu(batch):
    # 'batch' : [{dict}] (dict in list)
    keys = batch[0].keys()
    batched_data = {key: [item[key] for item in batch] for key in keys}

    batched_data["imu_data"] = pad_sequence(batched_data["imu_data"], batch_first=True)
    batched_data["dt"] = pad_sequence(batched_data["dt"], batch_first=True)

    # (img: torch.stack)
    for key in keys:
        if key != "imu_data":
            try:
                batched_data[key] = torch.stack(batched_data[key], dim=0)
            except:
                continue
    return batched_data


def main_custom():
    args = Config()
    args.settings_for_train_void_with_custom_network()

    log_dir = os.path.join(args.result_dir, 'train/logs')
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')
    logging.log_args_to_txt(log_txt, args)

    # Copy Config file
    shutil.copyfile(os.path.join(args.proj_dir, 'configs/config.py'),
                    os.path.join(args.result_dir, 'train/logs/config.py'))

    global result_dir
    result_dir = args.result_dir

    args.model_scale = 16
    model = IDEDepth(args=args)
    model.cuda()

    train_dataset = void_dataset_v2(cfg=args, data_path=args.customdata, crop_size=(args.crop_h, args.crop_w))
    val_dataset = void_dataset_v2(cfg=args, data_path=args.customdata, crop_size=(args.crop_h, args.crop_w), is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True, collate_fn=collate_fn_imu)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Training settings
    criterion_d = SiLogLoss()
    criterion_p = WeightedMSELoss(alpha=args.loss_weight_rot_to_trans_ratio)
    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                                             constructor='SwinLayerDecayOptimizerConstructor',
                                             paramwise_cfg=dict(num_layers=args.depths, layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    start_ep = 1

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    # Train & Validation
    best_rmse = 1000
    avg_losses = []
    for epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        avg_loss = train_custom(train_loader, model, criterion_d, criterion_p, log_txt,
                                optimizer=optimizer, epoch=epoch, args=args)
        avg_losses.append(avg_loss)
        writer.add_scalar('Training loss', avg_loss, epoch)

        # Save every model checkpoint
        if args.save_model:
            save_model(args, model, optimizer, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate_custom(val_loader, model, criterion_d, criterion_p,
                                                     epoch=epoch, args=args)
            writer.add_scalar('Val loss', loss_val, epoch)

            result_lines = logging.display_result(results_dict)
            print(result_lines)
            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)

            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch)

            # Save best model
            if args.save_model:
                best_rmse = save_model_best_rmse_model(args, model, optimizer, epoch, results_dict['rmse'], best_rmse)

    save_dir = os.path.join(result_dir, 'train')
    epochs = list(range(args.epochs))
    plt.figure()
    plt.plot(epochs, avg_losses, label='avg')
    plt.savefig(os.path.join(save_dir, 'Train_Losses.png'))


def train_custom(train_loader, model, criterion_d, criterion_p, log_txt, optimizer, epoch, args):
    global global_step
    model.train()
    depth_loss = logging.AverageMeter()
    pose_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    avg_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * (global_step / iterations / half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (global_step / iterations / half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr * param_group['lr_scale'] if 'swin' in args.backbone else current_lr

        frame1 = batch['frame1'].cuda()
        frame2 = batch['frame2'].cuda()
        depth_gt1 = batch['depth_image1'].cuda()
        depth_gt2 = batch['depth_image2'].cuda()
        rel_pose = batch['rel_pose'].cuda()

        preds = model(frame1, frame2)

        # Make the determinant of rotation matrix 1
        preds['out_p'][..., :9] = normalize_rot_vector(preds['out_p'][..., :9])
        rel_pose[..., :9] = normalize_rot_vector(rel_pose[..., :9])

        optimizer.zero_grad()
        loss_d1 = criterion_d(preds['pred_d1'].squeeze(1), depth_gt1)
        loss_d2 = criterion_d(preds['pred_d2'].squeeze(1), depth_gt2)
        loss_d = (loss_d1 + loss_d2) / 2
        loss_p = criterion_p(preds['out_p'], rel_pose)
        loss_t = loss_d + args.loss_weight_depth_to_pose_ratio * loss_p
        depth_loss.update(loss_d.item(), frame1.size(0))
        pose_loss.update(loss_p.item(), frame1.size(0))
        loss_t.backward()

        if args.pro_bar:
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                                 ('Depth Loss: %.4f (%.4f)\tPose Loss: %.4f (%.4f)' %
                                  (depth_loss.val, depth_loss.avg, pose_loss.val, pose_loss.avg)))

        if batch_idx % args.print_freq == 0:
            result_line = 'Epoch: [{0}][{1}/{2}]\tLoss_d: {loss_d}, Loss_p: {loss_p}, LR: {lr}\n'\
                .format(epoch, batch_idx, iterations, loss_d=depth_loss.avg, loss_p=pose_loss.avg, lr=current_lr)
            result_lines.append(result_line)
            print(result_line)
        optimizer.step()

        avg_loss += loss_t.item()

    with open(log_txt, 'a') as txtfile:
        txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        for result_line in result_lines:
            txtfile.write(result_line)
    avg_loss = avg_loss / iterations
    return avg_loss


def validate_custom(val_loader, model, criterion_d, criterion_p, epoch, args):
    depth_loss = logging.AverageMeter()
    model.eval()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    viz_tools = Visualize_CV()
    for batch_idx, batch in enumerate(val_loader):
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
        pred_d2 = preds['pred_d2'].squeeze()
        depth_gt1 = depth_gt1.squeeze()
        depth_gt2 = depth_gt2.squeeze()

        # Make the determinant of rotation matrix 1
        preds['out_p'][..., :9] = normalize_rot_vector(preds['out_p'][..., :9])
        rel_pose[..., :9] = normalize_rot_vector(rel_pose[..., :9])

        loss_d1 = criterion_d(pred_d1, depth_gt1)
        depth_loss.update(loss_d1.item(), frame1.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d1, depth_gt1)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        computed_mse = metrics.eval_pose(preds['out_p'], rel_pose)

        loss_d = depth_loss.avg

        for key in computed_result.keys():
            result_metrics[key] += computed_result[key]
        for key in computed_mse.keys():
            result_metrics[key] += computed_mse[key]

        if args.pro_bar:
            logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        if args.save_visualize:
            place, scene = filename.split('/')[-2:]
            save_dir = os.path.join(result_dir, 'test/depth', place)
            os.makedirs(save_dir, exist_ok=True)

            pred_d1_numpy = pred_d1.squeeze().cpu().numpy()
            pred_d1_numpy = (pred_d1_numpy / pred_d1_numpy.max()) * 255
            pred_d1_numpy = pred_d1_numpy.astype(np.uint8)
            pred_d1_color = cv2.applyColorMap(pred_d1_numpy, cv2.COLORMAP_RAINBOW)

            pred_d2_numpy = pred_d2.squeeze().cpu().numpy()
            pred_d2_numpy = (pred_d2_numpy / pred_d2_numpy.max()) * 255
            pred_d2_numpy = pred_d2_numpy.astype(np.uint8)
            pred_d2_color = cv2.applyColorMap(pred_d2_numpy, cv2.COLORMAP_RAINBOW)

            depth_gt1_numpy = depth_gt1.squeeze().cpu().numpy()
            depth_gt1_numpy = (depth_gt1_numpy / depth_gt1_numpy.max()) * 255
            depth_gt1_numpy = depth_gt1_numpy.astype(np.uint8)
            depth_gt1color = cv2.applyColorMap(depth_gt1_numpy, cv2.COLORMAP_RAINBOW)

            depth_gt2_numpy = depth_gt2.squeeze().cpu().numpy()
            depth_gt2_numpy = (depth_gt2_numpy / depth_gt2_numpy.max()) * 255
            depth_gt2_numpy = depth_gt2_numpy.astype(np.uint8)
            depth_gt2color = cv2.applyColorMap(depth_gt2_numpy, cv2.COLORMAP_RAINBOW)

            error_map1_numpy = np.abs(depth_gt1_numpy - pred_d1_numpy)
            error_map1_color = cv2.applyColorMap(error_map1_numpy, cv2.COLORMAP_WINTER)

            error_map2_numpy = np.abs(depth_gt2_numpy - pred_d2_numpy)
            error_map2_color = cv2.applyColorMap(error_map2_numpy, cv2.COLORMAP_WINTER)
            viz_tools.save_results(raw_image1, raw_image2,
                                   depth_gt1color, depth_gt2color,
                                   pred_d1_color, pred_d2_color,
                                   error_map1_color, error_map2_color,
                                   save_dir, scene)

            pos_save_dir = os.path.join(result_dir, 'test/pose', place)
            viz_tools.save_pos_results(rel_pose, preds['out_p'], pos_save_dir, scene)

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics, loss_d


if __name__ == '__main__':
    main_custom()
