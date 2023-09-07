# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2

from dataset.base_dataset_v2 import BaseDataset_v2
import torch
import pickle
import json
import numpy as np
import random
import subprocess
import pandas
from PIL import Image
from glob import glob
from utils.transformation_utils import exp_so3, inv_SE3_T_R, GetRelPose_tail2tail, log_SO3

def load_depth(path):
    '''
    Loads a depth map from a 16-bit PNG file

    Args:
    path : str
        path to 16-bit PNG file

    Returns:
    numpy : depth map
    '''
    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)
    # Assert 16-bit (not 8-bit) depth map
    z = z/256.0
    z[z <= 0] = 0.0
    return z

def load_pose(path):
    pose = np.loadtxt(path)
    return np.vstack([pose, [0, 0, 0, 1]])

def undistortion_image(image, K_intrinsic, camera_param):
    h, w = image.shape[:2]
    imgshape = (w, h)

    distCoeffs = np.array([camera_param['k0'], camera_param['p1'], camera_param['p1'], camera_param['p2'], camera_param['k2']])

    distortion_mat, _ = cv2.getOptimalNewCameraMatrix(K_intrinsic, distCoeffs, imgshape, 0)
    return cv2.undistort(image, K_intrinsic, distCoeffs, None, distortion_mat)

def convert_ros_csv(path):
    input_ros_path = path
    output_csv_path = path[:path.rfind(".bag")] + ".csv"
    command = "rostopic echo -b "+ input_ros_path +" -p /camera/imu > " + output_csv_path
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if not result.returncode == 0:
        print("Error output:", result.stderr)

def load_imu_csv(path, bias_acc, bias_gyro, R_c_i):
    imu_data = pandas.read_csv(path)
    stamp_column = ['field.header.stamp']
    acc_columns  = ['field.linear_acceleration.x', 'field.linear_acceleration.y', 'field.linear_acceleration.z']
    gyro_columns = ['field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z']
    stamp_values = imu_data[stamp_column].values *1e-9
    acc_values   = imu_data[acc_columns].values
    gyro_values  = imu_data[gyro_columns].values

    acc_i  = acc_values - bias_acc  # n*4
    gyro_i = gyro_values - bias_gyro  # n*4
    acc_c  = np.dot(R_c_i, acc_i.transpose())
    gyro_c = np.dot(R_c_i, gyro_i.transpose())
    return {'timestamp':stamp_values, 'acc':acc_c.transpose(), 'gyro':gyro_c.transpose()}

def get_imu_interval(imu_data, timestamp1, timestamp2):
    index = (imu_data['timestamp'] >= timestamp1) & (imu_data['timestamp'] <= timestamp2)
    index_3 = np.concatenate([index, index, index], axis=1)
    interval_timestamp = imu_data['timestamp'][index]
    interval_timestamp2 = interval_timestamp.copy()
    interval_timestamp2[1:] = interval_timestamp2[:-1]
    interval_timestamp2[0]  = timestamp1

    interval_dt   = (interval_timestamp - interval_timestamp2).reshape(-1,1)
    interval_acc  = imu_data['acc'][index_3].reshape(-1,3)
    interval_gyro = imu_data['gyro'][index_3].reshape(-1,3)
    imu_interval = np.concatenate([interval_dt, interval_acc, interval_gyro], axis=1)
    return interval_timestamp.reshape(-1,1), imu_interval

def check_imu_csv(raw_path):
    folders = sorted(os.listdir(raw_path))
    for folder  in folders:
        raw_folder_path = raw_path + folder + "/"
        # load list of files_path 
        rosbag_path = raw_folder_path + "/raw.bag"   #f or imu
        imu_csv_path = raw_folder_path + "/raw.csv"
        if not os.path.isfile(imu_csv_path):
            convert_ros_csv(rosbag_path)
            if os.path.isfile(rosbag_path):
                os.remove(rosbag_path)
            if os.path.isfile(raw_folder_path+"/dataset"):
                os.remove(raw_folder_path+"/dataset")
            if os.path.isfile(raw_folder_path+"/dataset_1500"):
                os.remove(raw_folder_path+"/dataset_1500")
            if os.path.isfile(raw_folder_path+"/dataset_500"):
                os.remove(raw_folder_path+"/dataset_500")

def get_relative_pose(RT01, RT02):
    RT12 = GetRelPose_tail2tail(RT01, RT02)
    T12 = RT12[:3,3].reshape(3,1)
    R12 = RT12[:3,:3]
    w12 = log_SO3(R12).reshape(3,1)
    AxisAngle12 = np.concatenate((w12/np.linalg.norm(w12), np.array([[np.linalg.norm(w12)]])), axis=0).reshape(4,1)

    return RT12, T12, R12, w12, AxisAngle12

class void_dataset_v3(BaseDataset_v2):
    def __init__(self, cfg, data_path,
                 is_train=True, crop_size=(448, 576), transform=False, image_interval_range=[5,5]):
        super().__init__(crop_size)
        
        # ------------------------------------------------------------------------------
        # <data_base_dir>
        #   |__void_raw
        #       |__birthplace_of_internet
        #       |__cabinet0
        #       |__ ...
        #   |__void_release
        #       |__void_1500
        #           |__data
        #               |__birthplace_of_internet
        #               |__cabinet0
        #               |__ ...
        # ------------------------------------------------------------------------------
        self.data_base_dir = data_path
        self.release_path = self.data_base_dir + "/void_release/void_1500/data/"
        self.raw_path = self.data_base_dir + "/void_raw/"
        
        #self.cfg = cfg
        self.is_train = is_train

        self.transform = transform
        self.image_interval_range = image_interval_range

        self.data_info_dir = os.path.dirname(os.path.abspath(__file__)) + "/void_dataset/"
        
        if is_train:
            self.data_path_file = os.path.join(self.data_info_dir, 'train_image.txt')
        else:
            self.data_path_file = os.path.join(self.data_info_dir, 'test_image.txt')

        with open(self.data_path_file) as f:
            self.file_path_list = [line.strip() for line in f]

        self.data_list = []
        self.data_preprocess()
        phase = 'train' if is_train else 'test'
        print("Dataset: VOID")
        print("# of %s images: %d" % (phase, len(self.data_list)))

        self.init_params()
        check_imu_csv(self.raw_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image1_path = self.data_list[idx]['file_path']
        folder_name = self.data_list[idx]['folder_name']
        release_folder_path = self.release_path + folder_name + "/"
        raw_folder_path = self.raw_path + folder_name + "/"
        
        imu_csv_path = raw_folder_path + "/raw.csv"
        imu_data = load_imu_csv(imu_csv_path, self.bias['acc'], self.bias['gyro'], self.calib['R_c_i'])
        
        image_list = glob(release_folder_path + "/image/*.png")
        depth_list = glob(release_folder_path + "/ground_truth/*.png")
        pose_list = glob(release_folder_path + "/absolute_pose/*.txt")
        image_list.sort()
        depth_list.sort()
        pose_list.sort()

        K_intrinsic = np.loadtxt(release_folder_path + "/K.txt", dtype=np.float64)
        
        order1 = self.data_list[idx]['order']
        order2 = order1 + random.randint(self.image_interval_range[0], self.image_interval_range[1])


        image1_stamp = float(image_list[order1][image_list[order1].rfind("/")+1:image_list[order1].rfind(".png")])
        image1 = cv2.imread(image_list[order1])
        depth1 = load_depth(depth_list[order1])
        image1_undistort = undistortion_image(image1, K_intrinsic, self.camera_param)
        depth1_undistort = undistortion_image(depth1, K_intrinsic, self.camera_param)
        RT01  = load_pose(pose_list[order1])
        
        image2_stamp = float(image_list[order2][image_list[order2].rfind("/")+1:image_list[order2].rfind(".png")])
        image2 = cv2.imread(image_list[order2])
        depth2 = load_depth(depth_list[order2])
        image2_undistort = undistortion_image(image2, K_intrinsic, self.camera_param)
        depth2_undistort = undistortion_image(depth2, K_intrinsic, self.camera_param)
        RT02  = load_pose(pose_list[order2])

        RT12, T12, R12, w12, AxisAngle12 = get_relative_pose(RT01, RT02)
        RT21, T21, R21, w21, AxisAngle21 = get_relative_pose(RT02, RT01)

        interval_timestamp, imu_interval = get_imu_interval(imu_data, image1_stamp, image2_stamp)

        if self.transform:
            # imu noise
            noise = torch.normal(mean=0, std=0.2, size=(imu_data.shape[0], imu_data.shape[1]))
            imu_data = imu_data + noise

            # image distortion
            # h1, w1 = raw_image1.shape[:2]
            # h2, w2 = raw_image2.shape[:2]
            # imgshape = (w1, h1)
            # distortion_mat, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, imgshape, 0)
            # undistorted_raw1 = cv2.undistort(raw_image1, self.cameraMatrix, self.distCoeffs, None, distortion_mat)
            # undistorted_raw2 = cv2.undistort(raw_image2, self.cameraMatrix, self.distCoeffs, None, distortion_mat)

            # undistorted_depth1 = cv2.undistort(depth_image1, self.cameraMatrix, self.distCoeffs, None, distortion_mat)
            # undistorted_depth2 = cv2.undistort(depth_image2, self.cameraMatrix, self.distCoeffs, None, distortion_mat)

            # undistorted_raw1 = cv2.cvtColor(undistorted_raw1, cv2.COLOR_BGR2RGB)
            # undistorted_raw2 = cv2.cvtColor(undistorted_raw2, cv2.COLOR_BGR2RGB)

        depth1 = depth1 / 1000.0  # convert in meters # 0.0010000000474974513
        depth2 = depth2 / 1000.0  # convert in meters
        depth1_undistort = depth1_undistort / 1000.0  # convert in meters # 0.0010000000474974513
        depth2_undistort = depth2_undistort / 1000.0  # convert in meters

        if self.is_train:
            # flip = random.randint(0, 1)
            # if flip == 1:
            #     undistorted_raw1, depth_image1 = self.do_flip(undistorted_raw1), self.do_flip(depth_image1)
            #     undistorted_raw2, depth_image2 = self.do_flip(undistorted_raw2), self.do_flip(depth_image2)
            image1_undistort, depth1_undistort = self.augment_training_data(image1_undistort, depth1_undistort)
            image2_undistort, depth2_undistort = self.augment_training_data(image2_undistort, depth2_undistort)
        else:
            image1_undistort, depth1_undistort = self.augment_test_data(image1_undistort, depth1_undistort)
            image2_undistort, depth2_undistort = self.augment_test_data(image2_undistort, depth2_undistort)
        
        image1 = self.to_tensor(image1).type(torch.FloatTensor)
        image2 = self.to_tensor(image2).type(torch.FloatTensor)
        depth1 = self.to_tensor(depth1).type(torch.FloatTensor)
        depth2 = self.to_tensor(depth2).type(torch.FloatTensor)
        T12 = self.to_tensor(T12).type(torch.FloatTensor).squeeze(0)
        R12 = self.to_tensor(R12).type(torch.FloatTensor).squeeze(0)
        w12 = self.to_tensor(w12).type(torch.FloatTensor).squeeze(0)
        AxisAngle12 = self.to_tensor(AxisAngle12).type(torch.FloatTensor).squeeze(0)
        T21 = self.to_tensor(T21).type(torch.FloatTensor).squeeze(0)
        R21 = self.to_tensor(R21).type(torch.FloatTensor).squeeze(0)
        w21 = self.to_tensor(w21).type(torch.FloatTensor).squeeze(0)
        AxisAngle21 = self.to_tensor(AxisAngle21).type(torch.FloatTensor).squeeze(0)
        interval_timestamp = self.to_tensor(interval_timestamp).type(torch.FloatTensor).squeeze(0)
        imu_interval = self.to_tensor(imu_interval).type(torch.FloatTensor).squeeze(0)

        input_data = {'filename':self.data_list[idx]['file_name'],
                'foldername':self.data_list[idx]['folder_name'],
                'timestamp1':image1_stamp,
                'timestamp2':image2_stamp,
                'image1': image1, 
                'image2': image2,
                'depth1': depth1, 
                'depth2': depth2,
                'image1_undistort': image1_undistort, 
                'image2_undistort': image2_undistort,
                'depth1_undistort': depth1_undistort, 
                'depth2_undistort': depth2_undistort,
                'T12': T12,
                'R12': R12,
                'w12': w12,
                'AxisAngle12': AxisAngle12,
                'T21': T21,
                'R21': R21,
                'w21': w21,
                'AxisAngle21': AxisAngle21,
                'imu_timestamp': interval_timestamp, 
                'imu_data': imu_interval, 
                }
        return input_data


    def init_params(self):
        self.calib = {}
        self.bias  = {}
        self.camera_param = {}
        calib_path = os.path.join(self.data_info_dir, 'calibration/calibration.json')
        with open(calib_path, 'r') as json_file:
            calib = json.load(json_file)
        self.calib['t_c_i'] = calib['alignment']['t_camera_to_body']   # 1*3
        self.calib['w_c_i'] = calib['alignment']['w_camera_to_body']   # 1*3, Rodrigues
        self.calib['R_c_i'] = exp_so3(self.calib['w_c_i'])
        self.calib['R_i_c'] = self.calib['R_c_i'].transpose()
        self.calib['RT_i_c'] = inv_SE3_T_R(self.calib['t_c_i'], self.calib['R_c_i'])
        
        # imu bias    
        self.bias['acc']  = calib['imu']['b_a']
        self.bias['gyro'] = calib['imu']['b_g']

        self.camera_param['k0'] = calib['camera']['k_0']
        self.camera_param['k1'] = calib['camera']['k_1']
        self.camera_param['k2'] = calib['camera']['k_2']
        self.camera_param['p1'] = calib['camera']['p_x']
        self.camera_param['p2'] = calib['camera']['p_y']

    def data_preprocess(self):
        for file_path in self.file_path_list:
            file_name = file_path[file_path.rfind('/')+1:]
            folder_name = file_path[file_path.rfind('/data/')+6 : file_path.rfind('/image/')]
            release_folder_path = self.release_path + folder_name + "/"
            image_list = sorted(os.listdir(release_folder_path+ "/image/"))

            order1 = image_list.index(file_name)
            order2 = order1 + self.image_interval_range[1]
            if order2 < len(image_list):
                data = {'file_path'   : file_path,
                        'file_name'   : file_name,
                        'folder_name' : folder_name,
                        'order'       : order1
                        }
                self.data_list.append(data)