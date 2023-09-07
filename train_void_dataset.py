# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.backup.model_ import GLPDepth
from models.optimizer import build_optimizers
import utils.metrics as metrics
from utils.criterion import SiLogLoss
import utils.logging as logging

from dataset.base_dataset import get_dataset
from configs.config import Config
from configs.train_options import TrainOptions
import glob

import pandas
from PIL import Image
import json
import pickle
from random import *

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None): # sequences : ['classroom0','classroom1', ...]

        self.transform = transform
        self.datapath = datapath

        self.raw_image1 = []
        self.raw_image2 = []
        self.depth_image1 = []
        self.depth_image2 = []
        self.rel_pose = []
        self.imu_data = []
        self.dt = []

        with open(self.datapath) as f: 
            self.file_paths = [line.strip() for line in f]
        # self.load_data() # when get dataset in seq (file folder set)


    def __len__(self): # num of dataset samples
        return len(self.file_paths)

    def __getitem__(self, idx):
        scan_path = self.file_paths[idx]
        with open(scan_path, 'rb') as f:
            raw_data = pickle.load(f)    
        
            raw_image1 = raw_data['raw_image1'][idx]
            raw_image2 =raw_data['raw_image2'][idx]
            depth_image1 = raw_data['depth_image1'][idx]
            depth_image2 = raw_data['depth_image2'][idx]
            rel_pose = raw_data['rel_pose'][idx]
            imu_data = raw_data['imu_data'][idx]
            dt = raw_data['dt'][idx]
            w = raw_data['Rodrigues'][idx]

        if self.transform:
            # imu noise
            noise = torch.normal(mean=0, std=0.2, size=(1, len(self.raw_image1))) 
            imu_data = imu_data + noise

        return raw_image1, raw_image2, depth_image1, depth_image2, rel_pose, imu_data, dt, w

    def load_data(self):
        self.data={}
        self.index={}
        data_idx = 0

        for seq in self.seq:
            datapath = os.path.join(self.data_path,seq,'customdataset')
            seq_data = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(datapath)) for f in fn]
            self.data[seq] = seq_data
            n_used_files = max(0, len(seq_data))
            for i in range(n_used_files):
                self.index[data_idx] = (seq,i)
                data_idx+=1


def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def main():
    """
    ==============================================================================
    Project path should be set as below
    ==============================================================================
    project_dir
      |__mim
          |__configs/config.py
          |__dataset/gen_data.py
          |__models/
          |__utils/
          |__weights/   [nyudepthv2_swin_base.ckpt, swin_v2_base_simmim.pth, swin_v2_large_simmim.pth]  pretrained backbone in here
          |__test.py
          |__train.py
          |__train_void_dataset.py   --> You are here ---> os.getcwd() returns [.../project_dir/mim] which should be equal with output path of line 148 (root_dir).
          |__train_custom.txt
          |__test_custom.txt
      |__void-dataset-master
          |__data/
              |__void_release/
                  |__tmp/
                  |__void_150/
                  |__void_500/
                  |__void_1500/
          |__scripts/
          |__bash/
          |__calibration/   [calibration.json, calibration.txt]  in here
          |__src/
      |__void_dataset_custom
    ==============================================================================
    """
    args = Config()

    args.settings_for_train_void_with_custom_network()
    args.gen_data = False

    root_dir = os.getcwd()

    log_dir = os.path.join(args.log_dir, args.exp_name)
    logging.check_and_make_dirs(log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = args.result_dir
    
    model = GLPDepth(args=args)

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    #### Dataset setting
    # calibration    
    calib = json.load(open(args.calib_path + '/calibration.json', 'r'))
    t_camera_to_body = calib['alignment']['t_camera_to_body'] #1*3
    w_camera_to_body = calib['alignment']['w_camera_to_body'] #1*3, Rodrigues
    theta = np.linalg.norm(w_camera_to_body)
    axis = w_camera_to_body / theta

    # Near phi==0, use first order Taylor expansion
    if np.abs(theta) < 1e-8:
        skew_phi = np.array([[0, -w_camera_to_body[2], w_camera_to_body[1]],
                    [w_camera_to_body[2], 0, -w_camera_to_body[0]],
                    [-w_camera_to_body[1], w_camera_to_body[0], 0]])
        R_camera_to_body = np.identity(3) + skew_phi
    else:
        w_hat = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
        R_camera_to_body = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * np.dot(w_hat, w_hat)

    T_camera_to_body = np.zeros((4, 4))
    T_camera_to_body[:3, :3] = np.linalg.inv(R_camera_to_body)
    T_camera_to_body[:3, 3:4] = np.dot(np.linalg.inv(R_camera_to_body),np.transpose(t_camera_to_body).reshape(3,1))
    T_camera_to_body[3:4, :3] = np.array([[0,0,0]])
    T_camera_to_body[3, 3] = 1

    # bias    
    imu_bias_acc = calib['imu']['b_a']
    imu_bias_gyro = calib['imu']['b_g']
    
    if args.gen_data:
        #### void dataset load, save
        folders = sorted(os.listdir(args.release_path))
        for folder in folders:
            dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.imu_path, 'void_release_path': args.release_path}
            dataset_kwargs['imu_csv'] = args.imu_path +'/'+ folder + '-raw-camera-imu.csv'
            dataset_kwargs['image_raw'] = args.release_path+'/'+ folder +'/image'
            dataset_kwargs['aligned_depth'] = args.release_path+'/'+ folder +'/ground_truth'
            dataset_kwargs['pose'] = args.release_path+'/'+ folder +'/absolute_pose'
            dataset_kwargs['output_save'] = '/media/ayoung/AY T5 1TB ntfs/void_dataset_custom/' + folder

            imu_data = pandas.read_csv(dataset_kwargs['imu_csv'])
            imu_data['.header.stamp.nsecs'] = imu_data['.header.stamp.nsecs'].apply(lambda x: round(int(str(x)[:4])))
            
            # Transform imu data to camera coordinate / imu bias applied
            imu_columns = ['.angular_velocity.x', '.angular_velocity.y', '.angular_velocity.z','.linear_acceleration.x', '.linear_acceleration.y', '.linear_acceleration.z']
            imu_values = imu_data[imu_columns].values
            gyro_homo = np.hstack((imu_values[:,0:3] - imu_bias_gyro, np.ones((imu_values.shape[0], 1)))) # n*4  
            acc_homo = np.hstack((imu_values[:,3:6] - imu_bias_acc, np.ones((imu_values.shape[0], 1)))) # n*4
            gyro_transformed_values = gyro_homo.dot(np.transpose(T_camera_to_body))  # n*4
            acc_transformed_values = acc_homo.dot(np.transpose(T_camera_to_body))

            # Update the original DataFrame with the transformed values
            imu_data[imu_columns] = pandas.DataFrame(np.hstack((gyro_transformed_values[:, 0:3], acc_transformed_values[:,0:3])), columns=imu_columns)
            
            # Read raw image & depth image
            raw_image_files = [f for f in os.listdir(dataset_kwargs['image_raw']) if f.endswith(".png")]
            depth_image_files = [f for f in os.listdir(dataset_kwargs['aligned_depth']) if f.endswith(".png")]

            # Image name ordering
            listup = sorted(raw_image_files)

            # Pick random image --- train set starts ---
            for idx, file in enumerate(raw_image_files):
                train_dataset=[]

                # raw & depth image read
                raw_image_path_1 = os.path.join(dataset_kwargs['image_raw'], file)
                depth_image_path_1 = os.path.join(dataset_kwargs['aligned_depth'], file)  # same name as raw image, preprocessing once!

                raw_image_1 = Image.open(raw_image_path_1)
                depth_image_1 = Image.open(depth_image_path_1)

                raw_image_tensor_1 = torch.tensor(np.array(raw_image_1))
                depth_image_tensor_1 = torch.tensor(np.array(depth_image_1))
                
                img_time_1 = float(file.split('.')[0]) + float(file.split('.')[1]) * 0.0001
                imu_time = imu_data['.header.stamp.secs'].values + imu_data['.header.stamp.nsecs'].values * 0.0001
                    

                # pose1 read
                pose1 = open(dataset_kwargs['pose'] + '/' + os.path.splitext(file)[0] + '.txt', 'r').read()
                # pose1_mat = [[float(num_str) for num_str in row.split()] for row in pose1.strip().split('\n')]
                # pose1_rot = [row[:3] for row in pose1_mat]
                # pose1_trans = [[row[3]] for row in pose1_mat]
                pose1_mat = np.fromstring(pose1, sep=' ').reshape(3, 4)
                pose1_transform = np.vstack([pose1_mat, [0, 0, 0, 1]])

                rand_idx = randint(10, 25)
                if img_time_1 is not None and (listup.index(file)-rand_idx) >= 0:

                    # raw & depth image read
                    raw_image_path_2 = os.path.join(dataset_kwargs['image_raw'], listup[listup.index(file)-rand_idx])  ## previous image
                    depth_image_path_2 = os.path.join(dataset_kwargs['aligned_depth'], listup[listup.index(file)-rand_idx]) # same name as raw image, preprocessing once!

                    raw_image_2 = Image.open(raw_image_path_2)
                    depth_image_2 = Image.open(depth_image_path_2)

                    raw_image_tensor_2 = torch.tensor(np.array(raw_image_2))
                    depth_image_tensor_2 = torch.tensor(np.array(depth_image_2))
                    
                    img_time_2 = float(listup[listup.index(file)-rand_idx].split('.')[0]) + float(listup[listup.index(file)-rand_idx].split('.')[1]) * 0.0001
                    
                    # imu data interval
                    imu_data_interval = imu_data[(imu_time >= img_time_2) & (imu_time <= img_time_1)] #img1-time2 / img2-time1
                    imu_interval_tensor = torch.tensor(imu_data_interval[imu_columns].values)

                    # dt
                    imu_interval_time = imu_data_interval['.header.stamp.secs'].values + imu_data_interval['.header.stamp.nsecs'].values * 0.0001
                    dt_sec = imu_interval_time - img_time_2

                    # pose2 read
                    pose2 = open(dataset_kwargs['pose'] + '/' + os.path.splitext(listup[listup.index(file)-rand_idx])[0] + '.txt', 'r').read()
                    # pose2_mat = [[float(num_str) for num_str in row.split()] for row in pose2.strip().split('\n')]
                    # pose2_rot = [row[:3] for row in pose2_mat]
                    # pose2_trans = [[row[3]] for row in pose2_mat]
                    pose2_mat = np.fromstring(pose2, sep=' ').reshape(3, 4)
                    pose2_transform = np.vstack([pose2_mat, [0, 0, 0, 1]])
                    relative_pose = np.dot(np.linalg.inv(pose2_transform),pose1_transform)
                    rvec, _ = cv2.Rodrigues(relative_pose[:3, :3])
                    rvec_tensor = torch.tensor(rvec)
                    relative_pose_tensor = torch.tensor(relative_pose)
                    
                    # train set
                    train_dataset.append({'raw_image1': raw_image_tensor_2,'raw_image2': raw_image_tensor_1, 
                                        'depth_image1': depth_image_tensor_2,'depth_image2': depth_image_tensor_1, 
                                        'rel_pose': relative_pose_tensor, 'imu_data': imu_interval_tensor,
                                        'dt': dt_sec, 'Rodrigues':rvec_tensor})
                    
                        
                    # Save train data
                    if not os.path.exists(dataset_kwargs['output_save']):
                        os.makedirs(dataset_kwargs['output_save'])
                    output_file_path = dataset_kwargs['output_save'] + '/' + file.split('.')[0] + '.' + file.split('.')[1] + ".pkl"
                    with open(output_file_path, 'wb') as output_file:
                        pickle.dump(train_dataset, output_file)


                else:
                    continue

    # read void train / test set
    with open(os.path.join(args.data_path, 'void_1500/train_image.txt'), 'r') as file:
        trainset = file.readlines()
    trainsets = [os.path.join(args.customdata, line.strip().split('/')[2], os.path.basename(line.strip())).replace('.png', '.pkl') for line in trainset]

    with open(os.path.join(args.data_path, 'void_1500/test_image.txt'), 'r') as file:
        testset = file.readlines()
    testsets = [os.path.join(args.customdata, line.strip().split('/')[2], os.path.basename(line.strip())).replace('.png', '.pkl') for line in testset]

    # make customset
    train_path = os.path.join(args.customdata, 'train_custom.txt')
    test_path = os.path.join(args.customdata, 'test_custom.txt')
    with open(train_path, 'w') as file:
        file.writelines(line + '\n' for line in trainsets)

    with open(test_path, 'w') as file:
        file.writelines(line + '\n' for line in testsets)


def main_custom():
    """
    ==============================================================================
    Project path should be set as below
    ==============================================================================
    project_dir
      |__mim
          |__configs/config.py
          |__dataset/gen_data.py
          |__models/
          |__utils/
          |__weights/   [nyudepthv2_swin_base.ckpt, swin_v2_base_simmim.pth, swin_v2_large_simmim.pth]  pretrained backbone in here
          |__test.py
          |__train.py
          |__train_void_dataset.py   --> You are here ---> os.getcwd() returns [.../project_dir/mim] which should be equal with output path of line 148 (root_dir).
          |__train_custom.txt
          |__test_custom.txt
      |__void-dataset-master
          |__data/
              |__void_release/
                  |__tmp/
                  |__void_150/
                  |__void_500/
                  |__void_1500/
          |__scripts/
          |__bash/
          |__calibration/   [calibration.json, calibration.txt]  in here
          |__src/
      |__void_dataset_custom
    ==============================================================================
    """
    args = Config()

    args.settings_for_train_void_with_custom_network()
    args.gen_data = True

    #### Dataset setting
    # calibration
    calib = json.load(open(args.calib_path + '/calibration.json', 'r'))
    t_camera_to_body = calib['alignment']['t_camera_to_body']  # 1*3
    w_camera_to_body = calib['alignment']['w_camera_to_body']  # 1*3, Rodrigues
    theta = np.linalg.norm(w_camera_to_body)
    axis = w_camera_to_body / theta

    # Near phi==0, use first order Taylor expansion
    if np.abs(theta) < 1e-8:
        skew_phi = np.array([[0, -w_camera_to_body[2], w_camera_to_body[1]],
                             [w_camera_to_body[2], 0, -w_camera_to_body[0]],
                             [-w_camera_to_body[1], w_camera_to_body[0], 0]])
        R_camera_to_body = np.identity(3) + skew_phi
    else:
        w_hat = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        R_camera_to_body = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * np.dot(w_hat, w_hat)

    T_camera_to_body = np.zeros((4, 4))
    T_camera_to_body[:3, :3] = np.linalg.inv(R_camera_to_body)
    T_camera_to_body[:3, 3:4] = np.dot(np.linalg.inv(R_camera_to_body), np.transpose(t_camera_to_body).reshape(3, 1))
    T_camera_to_body[3:4, :3] = np.array([[0, 0, 0]])
    T_camera_to_body[3, 3] = 1

    # bias
    imu_bias_acc = calib['imu']['b_a']
    imu_bias_gyro = calib['imu']['b_g']

    if args.gen_data:
        #### void dataset load, save
        folders = sorted(os.listdir(args.release_path))
        for folder in folders:
            dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.imu_path, 'void_release_path': args.release_path}
            dataset_kwargs['imu_csv'] = os.path.join(args.imu_path, folder + '-raw-camera-imu.csv')
            dataset_kwargs['image_raw'] = os.path.join(args.release_path, folder + '/image')
            dataset_kwargs['aligned_depth'] = os.path.join(args.release_path, folder + '/ground_truth')
            dataset_kwargs['pose'] = os.path.join(args.release_path, folder + '/absolute_pose')
            dataset_kwargs['output_save'] = os.path.join(args.customdata, folder)

            imu_data = pandas.read_csv(dataset_kwargs['imu_csv'])
            imu_data['.header.stamp.nsecs'] = imu_data['.header.stamp.nsecs'].apply(lambda x: round(int(str(x)[:4])))

            # Transform imu data to camera coordinate / imu bias applied
            imu_columns = ['.angular_velocity.x', '.angular_velocity.y', '.angular_velocity.z',
                           '.linear_acceleration.x', '.linear_acceleration.y', '.linear_acceleration.z']
            imu_values = imu_data[imu_columns].values
            gyro_homo = np.hstack((imu_values[:, 0:3] - imu_bias_gyro, np.ones((imu_values.shape[0], 1))))  # n*4
            acc_homo = np.hstack((imu_values[:, 3:6] - imu_bias_acc, np.ones((imu_values.shape[0], 1))))  # n*4
            gyro_transformed_values = gyro_homo.dot(np.transpose(T_camera_to_body))  # n*4
            acc_transformed_values = acc_homo.dot(np.transpose(T_camera_to_body))

            # Update the original DataFrame with the transformed values
            imu_data[imu_columns] = pandas.DataFrame(np.hstack((gyro_transformed_values[:, 0:3], acc_transformed_values[:, 0:3])), columns=imu_columns)

            # Read raw image & depth image
            raw_image_files = [f for f in os.listdir(dataset_kwargs['image_raw']) if f.endswith(".png")]
            depth_image_files = [f for f in os.listdir(dataset_kwargs['aligned_depth']) if f.endswith(".png")]

            # Image name ordering
            listup = sorted(raw_image_files)

            # Pick random image --- train set starts ---
            for idx, file in enumerate(raw_image_files):
                print(f'{folder}:{file}:{idx}/{len(raw_image_files)}', end='\r')
                train_dataset = []

                # raw & depth image read
                raw_image_path_1 = os.path.join(dataset_kwargs['image_raw'], file)
                depth_image_path_1 = os.path.join(dataset_kwargs['aligned_depth'], file)  # same name as raw image, preprocessing once!

                raw_image_1 = cv2.imread(raw_image_path_1)  # [480, 640, 3]
                # raw_image_1 = cv2.cvtColor(raw_image_1, cv2.COLOR_BGR2RGB)
                depth_image_1 = cv2.imread(depth_image_path_1, cv2.IMREAD_UNCHANGED).astype('float32')  # [480, 640]

                img_time_1 = float(file.split('.')[0]) + float(file.split('.')[1]) * 0.0001
                imu_time = imu_data['.header.stamp.secs'].values + imu_data['.header.stamp.nsecs'].values * 0.0001

                # pose1 read
                pose1 = open(dataset_kwargs['pose'] + '/' + os.path.splitext(file)[0] + '.txt', 'r').read()
                pose1_mat = np.fromstring(pose1, sep=' ').reshape(3, 4)
                pose1_transform = np.vstack([pose1_mat, [0, 0, 0, 1]])

                rand_idx = randint(10, 25)
                if img_time_1 is not None and (listup.index(file) - rand_idx) >= 0:

                    # raw & depth image read
                    raw_image_path_2 = os.path.join(dataset_kwargs['image_raw'], listup[listup.index(file) - rand_idx])  ## previous image
                    depth_image_path_2 = os.path.join(dataset_kwargs['aligned_depth'], listup[listup.index(file) - rand_idx])  # same name as raw image, preprocessing once!

                    raw_image_2 = cv2.imread(raw_image_path_2)  # [480, 640, 3]
                    # raw_image_2 = cv2.cvtColor(raw_image_2, cv2.COLOR_BGR2RGB)
                    depth_image_2 = cv2.imread(depth_image_path_2, cv2.IMREAD_UNCHANGED).astype('float32')  # [480, 640]

                    img_time_2 = float(listup[listup.index(file) - rand_idx].split('.')[0]) + \
                                 float(listup[listup.index(file) - rand_idx].split('.')[1]) * 0.0001

                    # imu data interval
                    imu_data_interval = imu_data[
                        (imu_time >= img_time_2) & (imu_time <= img_time_1)]  # img1-time2 / img2-time1
                    imu_interval_tensor = torch.tensor(imu_data_interval[imu_columns].values)

                    # dt
                    imu_interval_time = imu_data_interval['.header.stamp.secs'].values + \
                                        imu_data_interval['.header.stamp.nsecs'].values * 0.0001
                    dt_sec = imu_interval_time - img_time_2

                    # pose2 read
                    pose2 = open(dataset_kwargs['pose'] + '/' + os.path.splitext(listup[listup.index(file) - rand_idx])[0] + '.txt', 'r').read()

                    pose2_mat = np.fromstring(pose2, sep=' ').reshape(3, 4)
                    pose2_transform = np.vstack([pose2_mat, [0, 0, 0, 1]])
                    relative_pose = np.dot(np.linalg.inv(pose2_transform), pose1_transform)
                    rvec, _ = cv2.Rodrigues(relative_pose[:3, :3])
                    rvec_tensor = torch.tensor(rvec)
                    relative_pose_tensor = torch.tensor(relative_pose)

                    # train set
                    train_dataset.append({'raw_image1': raw_image_1, 'raw_image2': raw_image_2,
                                          'depth_image1': depth_image_1, 'depth_image2': depth_image_2,
                                          'rel_pose': relative_pose_tensor, 'imu_data': imu_interval_tensor,
                                          'dt': dt_sec, 'Rodrigues': rvec_tensor})

                    # Save train data
                    if not os.path.exists(dataset_kwargs['output_save']):
                        os.makedirs(dataset_kwargs['output_save'])
                    output_file_path = dataset_kwargs['output_save'] + '/' + file.split('.')[0] + '.' + file.split('.')[1] + ".pkl"
                    with open(output_file_path, 'wb') as output_file:
                        pickle.dump(train_dataset, output_file)

                else:
                    continue

    # read void train / test set
    with open(os.path.join(args.data_path, 'void_1500/train_image.txt'), 'r') as file:
        trainset = file.readlines()
    trainsets = [os.path.join(args.customdata,
                              line.strip().split('/')[2],
                              os.path.basename(line.strip())).replace('.png', '.pkl') for line in trainset]

    with open(os.path.join(args.data_path, 'void_1500/test_image.txt'), 'r') as file:
        testset = file.readlines()
    testsets = [os.path.join(args.customdata,
                             line.strip().split('/')[2],
                             os.path.basename(line.strip())).replace('.png', '.pkl') for line in testset]

    # make customset
    train_path = os.path.join(args.customdata, 'train_custom.txt')
    test_path = os.path.join(args.customdata, 'test_custom.txt')
    with open(train_path, 'w') as file:
        file.writelines(line + '\n' for line in trainsets)

    with open(test_path, 'w') as file:
        file.writelines(line + '\n' for line in testsets)


if __name__ == '__main__':
    # main()
    main_custom()
