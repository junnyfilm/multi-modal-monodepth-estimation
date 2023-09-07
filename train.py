# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import torch
import yaml
import shutil
import cv2
import numpy as np
import argparse
from time import time
import glob

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

import utils.metrics as metrics
import utils.logging as logging

import torch.backends.cudnn as cudnn

from configs.config import Config
from models.model import IDEDepth
from models.optimizer import build_optimizers
from dataset.void_dataset_v3 import void_dataset_v3
from utils.criterion import SiLogLoss, WeightedMSELoss
from utils.util import save_model, save_model_best_rmse_model, load_model
from torch.nn.utils.rnn import pad_sequence

from utils.viz_utils import Visualize_CV

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--config', type=str, default='config.yaml', help='config yaml file name')


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 
               'pose_mse_r12', 'pose_mse_t12','pose_mse_r21', 'pose_mse_t21',
               'pose_mse_r_identity', 'pose_mse_t_identity'
               ]

def collate_fn_imu(batch):
    keys = batch[0].keys()
    batched_data = {key: [item[key] for item in batch] for key in keys}
    
    batched_data["imu_data"] = [data.clone() for data in batched_data["imu_data"]]
    batched_data["imu_data"] = pad_sequence(batched_data["imu_data"], batch_first=True, padding_value=0)

    batched_data["imu_timestamp"] = [data.clone() for data in batched_data["imu_timestamp"]]
    batched_data["imu_timestamp"] = pad_sequence(batched_data["imu_timestamp"], batch_first=True, padding_value=0)

    for key in keys:
        if key != "imu_data":
            try:
                batched_data[key] = torch.stack(batched_data[key], dim=0)
            except:
                continue
    return batched_data

def main():
    opt = parser.parse_args()
    CONFIG_YAML_NAME = opt.config
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_yaml_path = BASE_DIR + "/configs/" + CONFIG_YAML_NAME
    with open(config_yaml_path, "r") as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        print(config_yaml)

    args = Config(config_yaml)
    if args.resume_from:
        args.log_dir = args.resume_from[:args.resume_from.rfind("/train/ckpt")]
        writer = SummaryWriter(logdir=args.log_dir)
        log_txt = os.path.join(args.log_dir, 'logs.txt')
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(logdir=args.log_dir)
        log_txt = os.path.join(args.log_dir, 'logs.txt')
        logging.log_args_to_txt(log_txt, args)
        shutil.copyfile(config_yaml_path, os.path.join(args.log_dir, 'config.yaml'))    

    # writer = SummaryWriter(logdir=args.log_dir)
    # log_txt = os.path.join(args.log_dir, 'logs.txt')
    # logging.log_args_to_txt(log_txt, args)
    # shutil.copyfile(config_yaml_path, os.path.join(args.log_dir, 'config.yaml'))

    model = IDEDepth(args=args)

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        device_count = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    train_dataset = void_dataset_v3(cfg=args, data_path=args.data_dir, crop_size=(args.crop_h, args.crop_w), image_interval_range=args.image_interval_range)
    val_dataset = void_dataset_v3(cfg=args, data_path=args.data_dir, crop_size=(args.crop_h, args.crop_w), is_train=False, image_interval_range=args.image_interval_range)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True, collate_fn = collate_fn_imu)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Training settings
    criterion_d = SiLogLoss()
    criterion_p = WeightedMSELoss()
    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='SwinLayerDecayOptimizerConstructor',
                paramwise_cfg=dict(num_layers=args.depths, layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    start_ep = 1

    if args.resume_from:
        resume_ep = load_model(args.resume_from, model.module, optimizer)
        print(f'resumed from epoch {resume_ep}, ckpt {args.resume_from}')
        start_ep = resume_ep + 1
    # if args.auto_resume and not args.resume_from:
    #     ckpt_list = glob.glob(args.log_dir+ "train/ckpt/epoch_*_model.ckpt")
    #     ckpt_list.sort()
    #     if len(ckpt_list) > 0:
    #         load_model(ckpt_list[-1], model.module, optimizer)
    #         strlength = len('_model.ckpt')
    #         resume_ep = int(ckpt_list[-1][-strlength-2:-strlength])
    #         print(f'resumed from epoch {resume_ep}, ckpt {ckpt_list[-1]}')
    #         start_ep = resume_ep + 1
    
    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    # Train & Validation
    best_rmse = 1000
    avg_losses_total = []
    for epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        avg_loss_total, avg_loss_depth, avg_loss_traslation, avg_loss_rotation = train(train_loader, model, criterion_d, criterion_p, log_txt,
                                optimizer=optimizer, device=device, epoch=epoch, args=args)
        writer.add_scalar('Training loss total', avg_loss_total, epoch)
        writer.add_scalar('Training loss depth', avg_loss_depth, epoch)
        writer.add_scalar('Training loss translation', avg_loss_traslation, epoch)
        writer.add_scalar('Training loss rotation', avg_loss_rotation, epoch)

        avg_losses_total.append(avg_loss_total)

        # Save every model checkpoint
        if args.save_model:
            save_model(args, model, optimizer, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val_d, loss_val_T, loss_val_R = validate(val_loader, model, criterion_d, criterion_p,
                                                    device=device, epoch=epoch, args=args)
            writer.add_scalar('Val loss depth', loss_val_d, epoch)
            writer.add_scalar('Val loss translation', loss_val_T, epoch)
            writer.add_scalar('Val loss rotation', loss_val_R, epoch)

            result_lines = logging.display_result(results_dict)
            print(result_lines)
            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)

            # Save best model
            if args.save_model:
                best_rmse = save_model_best_rmse_model(args, model, optimizer, epoch, results_dict['rmse'], best_rmse)
            
            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch)
    epochs = list(range(len(avg_losses_total)))
    plt.figure()
    plt.plot(epochs, avg_losses_total, label='avg')
    plt.savefig(os.path.join(args.log_dir, "Train_Losses.png"))


def train(train_loader, model, criterion_d, criterion_p, log_txt, optimizer, device, epoch, args):
    global global_step
    model.train()
    depth_loss = logging.AverageMeter()
    translation_loss = logging.AverageMeter()
    rotation_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    avg_loss_total = 0
    now_time = time()
    for batch_idx, batch in enumerate(train_loader):
        #torch.cuda.empty_cache()
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * \
                        (global_step / iterations / half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * \
                        (global_step / iterations / half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr * param_group['lr_scale'] if 'swin' in args.backbone else current_lr

        image1 = batch['image1_undistort'].to(device)
        image2 = batch['image2_undistort'].to(device)
        depth1_gt = batch['depth1_undistort'].to(device)
        depth2_gt = batch['depth2_undistort'].to(device)
        T12_gt = batch['T12'].to(device)
        R12_gt = batch['R12'].to(device)
        T21_gt = batch['T21'].to(device)
        R21_gt = batch['R21'].to(device)
        preds = model(image1, image2)
        optimizer.zero_grad()
        loss_d1 = criterion_d(preds['pred_d1'].squeeze(1), depth1_gt)
        loss_d2 = criterion_d(preds['pred_d2'].squeeze(1), depth2_gt)
        loss_depth = (loss_d1 + loss_d2) / 2
        if args.decoder == "decoder_v1":
            loss_R12 = criterion_p(preds['pred_r12'], R12_gt)
            loss_T12 = criterion_p(preds['pred_t12'], T12_gt)
            loss_Rotation = loss_R12
            loss_Translation = loss_R12
        elif args.decoder == "decoder_v2":
            loss_R12 = criterion_p(preds['pred_r12'], R12_gt)
            loss_T12 = criterion_p(preds['pred_t12'], T12_gt)
            loss_R21 = criterion_p(preds['pred_r21'], R21_gt)
            loss_T21 = criterion_p(preds['pred_t21'], T21_gt)
            loss_Rotation = (loss_R12 + loss_R21) / 2
            loss_Translation = (loss_T12 + loss_T21) / 2
        loss_total = loss_depth + args.loss_lambda1 * loss_Rotation + args.loss_lambda2 * loss_Translation
        depth_loss.update(loss_depth.item(), image1.size(0))
        translation_loss.update(loss_Translation.item(), image1.size(0))
        rotation_loss.update(loss_Rotation.item(), image1.size(0))

        loss_total.backward()

        if args.pro_bar:
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                                 ('Depth Loss: %.4f (%.4f)\tTranslation Loss: %.4f (%.4f)\tRotation Loss: %.4f (%.4f)' %
                                  (depth_loss.val, depth_loss.avg, translation_loss.val, translation_loss.avg, rotation_loss.val, rotation_loss.avg,)))
        if batch_idx % args.print_freq == 0:
            result_line = 'Epoch: [{0}][{1}/{2}]\tLoss_d: {loss_d}, Loss_T: {Loss_T}, Loss_R: {Loss_R}, LR: {lr}\n'\
                .format(epoch, batch_idx, iterations, loss_d=depth_loss.avg, Loss_T=translation_loss.avg, Loss_R=rotation_loss.avg, lr=current_lr)
            result_lines.append(result_line)
            print(result_line[:-2])
        optimizer.step()
        avg_loss_total += loss_total.item()
        print("[Time for 1 Iter] :", time() - now_time)
        now_time = time()
        
    with open(log_txt, 'a') as txtfile:
        txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        for result_line in result_lines:
            txtfile.write(result_line)
    avg_loss_total = avg_loss_total / iterations # To-do list

    return avg_loss_total, depth_loss.avg, translation_loss.avg, rotation_loss.avg


def validate(val_loader, model, criterion_d, criterion_p, device, epoch, args):
    now_time = time()

    depth_loss = logging.AverageMeter()
    translation_loss = logging.AverageMeter()
    rotation_loss = logging.AverageMeter()
    model.eval()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    viz_tools = Visualize_CV()
    for batch_idx, batch in enumerate(val_loader):
        image1 = batch['image1_undistort'].to(device)
        image2 = batch['image2_undistort'].to(device)
        depth1_gt = batch['depth1_undistort'].to(device)
        depth2_gt = batch['depth2_undistort'].to(device)
        T12_gt = batch['T12'].to(device)
        R12_gt = batch['R12'].to(device)
        T21_gt = batch['T21'].to(device)
        R21_gt = batch['R21'].to(device)

        with torch.no_grad():
            preds = model(image1, image2)

        pred_d1 = preds['pred_d1'].squeeze()
        pred_d2 = preds['pred_d2'].squeeze()
        depth1_gt = depth1_gt.squeeze()
        depth2_gt = depth2_gt.squeeze()

        loss_d1 = criterion_d(pred_d1, depth1_gt)
        loss_d2 = criterion_d(pred_d2, depth2_gt)
        loss_depth = (loss_d1 + loss_d2) / 2
        if args.decoder == "decoder_v1":
            loss_R12 = criterion_p(preds['pred_r12'], R12_gt)
            loss_T12 = criterion_p(preds['pred_t12'], T12_gt)
            loss_Rotation = loss_R12
            loss_Translation = loss_R12
        elif args.decoder == "decoder_v2":
            loss_R12 = criterion_p(preds['pred_r12'], R12_gt)
            loss_T12 = criterion_p(preds['pred_t12'], T12_gt)
            loss_R21 = criterion_p(preds['pred_r21'], R21_gt)
            loss_T21 = criterion_p(preds['pred_t21'], T21_gt)
            loss_Rotation = (loss_R12 + loss_R21) / 2
            loss_Translation = (loss_T12 + loss_T21) / 2

        depth_loss.update(loss_depth.item(), image1.size(0))
        translation_loss.update(loss_Translation.item(), image1.size(0))
        rotation_loss.update(loss_Rotation.item(), image1.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d1, depth1_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        gt_pose = { "R12": R12_gt,
                    "T12": T12_gt,
                    "R21": R21_gt,
                    "T21": T21_gt,
                    }
        pred_pose = { "R12": preds['pred_r12'],
                    "T12": preds['pred_t12'],
                    "R21": preds['pred_r21'],
                    "T21": preds['pred_t21'],
                    }
        computed_mse = metrics.eval_pose(pred_pose, gt_pose)
        # save_path = os.path.join(result_dir, filename)

        # if save_path.split('.')[-1] == 'jpg':
        #     save_path = save_path.replace('jpg', 'png')
        #
        # if args.save_result:
        #     if args.dataset == 'kitti':
        #         pred_d_numpy = pred_d.cpu().numpy() * 256.0
        #         cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #     else:
        #         pred_d_numpy = pred_d.cpu().numpy() * 1000.0
        #         cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loss_d = depth_loss.avg
        loss_T = translation_loss.avg
        loss_R = rotation_loss.avg
        if args.pro_bar:
            logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        if args.save_visualize:
            filename = batch['filename'][0]
            foldername = batch['foldername'][0]
            save_dir = os.path.join(args.log_dir, 'test/depth', foldername, str(epoch))
            os.makedirs(save_dir, exist_ok=True)

            image1_view = image1.squeeze().cpu().numpy()
            image1_view = np.transpose(image1_view, (1, 2, 0)) * 255

            image2_view = image2.squeeze().cpu().numpy()
            image2_view = np.transpose(image2_view, (1, 2, 0)) * 255

            pred_d1_numpy   = pred_d1.squeeze().cpu().numpy()
            pred_d2_numpy   = pred_d2.squeeze().cpu().numpy()
            depth1_gt_numpy = depth1_gt.squeeze().cpu().numpy()
            depth2_gt_numpy = depth2_gt.squeeze().cpu().numpy()
            
            depth_max = max([pred_d1_numpy.max(), pred_d2_numpy.max()])
            pred_d1_numpy   = (pred_d1_numpy / depth_max) * 255
            pred_d2_numpy   = (pred_d2_numpy / depth_max) * 255
            depth1_gt_numpy = (depth1_gt_numpy / depth_max) * 255
            depth2_gt_numpy = (depth2_gt_numpy / depth_max) * 255

            pred_d1_numpy   = pred_d1_numpy.astype(np.uint8)
            pred_d2_numpy   = pred_d2_numpy.astype(np.uint8)
            depth1_gt_numpy = depth1_gt_numpy.astype(np.uint8)
            depth2_gt_numpy = depth2_gt_numpy.astype(np.uint8)

            pred_d1_color   = cv2.applyColorMap(pred_d1_numpy, cv2.COLORMAP_RAINBOW)
            pred_d2_color   = cv2.applyColorMap(pred_d2_numpy, cv2.COLORMAP_RAINBOW)
            depth1_gt_color = cv2.applyColorMap(depth1_gt_numpy, cv2.COLORMAP_RAINBOW)
            depth2_gt_color = cv2.applyColorMap(depth2_gt_numpy, cv2.COLORMAP_RAINBOW)

            error_map1_numpy = np.abs(depth1_gt_numpy - pred_d1_numpy)
            error_map1_color = cv2.applyColorMap(error_map1_numpy, cv2.COLORMAP_WINTER)

            error_map2_numpy = np.abs(depth2_gt_numpy - pred_d2_numpy)
            error_map2_color = cv2.applyColorMap(error_map2_numpy, cv2.COLORMAP_WINTER)
            viz_tools.save_results(image1_view, image2_view,
                                   depth1_gt_color, depth2_gt_color,
                                   pred_d1_color, pred_d2_color,
                                   error_map1_color, error_map2_color,
                                   save_dir, filename)

            pos_save_dir = os.path.join(args.log_dir, 'test/pose', foldername, str(epoch))
            os.makedirs(pos_save_dir, exist_ok=True)
            viz_tools.save_pos_results(R12_gt, preds['pred_r12'], T12_gt, preds['pred_t12'], pos_save_dir, filename)
        
        for key in computed_result.keys():
            result_metrics[key] += computed_result[key]
        for key in computed_mse.keys():
            result_metrics[key] += computed_mse[key]
    
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)
    print("[Time for Validation] :", time() - now_time)
    return result_metrics, loss_d, loss_T, loss_R


if __name__ == '__main__':
    main()
