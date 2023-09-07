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

class void_dataset_v2(BaseDataset_v2):
    def __init__(self, cfg, data_path,
                 is_train=True, crop_size=(448, 576), transform=True, scale_size=None):
        super().__init__(crop_size)
        self.cfg = cfg
        self.get_camera_params()

        self.is_train = is_train
        self.data_path = data_path

        self.scale_size = scale_size
        self.transform = transform

        self.raw_image1 = []
        self.raw_image2 = []
        self.depth_image1 = []
        self.depth_image2 = []
        self.rel_pose = []
        self.imu_data = []
        self.dt = []

        # txt_path = os.path.join(filenames_path, 'nyudepthv2')
        if is_train:
            self.txt_folder = os.path.join(data_path, 'train_custom.txt')
        else:
            self.txt_folder = os.path.join(data_path, 'test_custom.txt')
            # self.data_path = self.data_path + '/official_splits/test/'

        with open(self.txt_folder) as f:
            self.filenames_list = [line.strip() for line in f]
        # with open(self.data_path) as f:
        #     self.filenames_list = [line.strip() for line in f]
        phase = 'train' if is_train else 'test'
        print("Dataset: VOID")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        scan_path = self.filenames_list[idx]
        with open(scan_path, 'rb') as f:
            raw_data = pickle.load(f)

        raw_image1 = raw_data[0]['raw_image1']
        raw_image2 = raw_data[0]['raw_image2']
        depth_image1 = raw_data[0]['depth_image1']
        depth_image2 = raw_data[0]['depth_image2']
        undistorted_raw1 = raw_data[0]['undistorted_raw1']
        undistorted_raw2 = raw_data[0]['undistorted_raw2']
        undistorted_depth1 = raw_data[0]['undistorted_depth1']
        undistorted_depth2 = raw_data[0]['undistorted_depth2']
        rel_pose = raw_data[0]['rel_pose']
        imu_data = raw_data[0]['imu_data']
        dt = raw_data[0]['dt']
        w = raw_data[0]['Rodrigues']

        if self.transform:
            # imu noise
            noise = torch.normal(mean=0, std=0.2, size=(imu_data.shape[0], imu_data.shape[1]))
            imu_data = imu_data + noise

            # image distortion
            h1, w1 = raw_image1.shape[:2]
            h2, w2 = raw_image2.shape[:2]
            imgshape = (w1, h1)

        # depth_image1 = depth_image1 / 1000.0  # convert in meters
        # depth_image2 = depth_image2 / 1000.0  # convert in meters
        undistorted_depth1 = undistorted_depth1 / 1000.0  # convert in meters
        undistorted_depth2 = undistorted_depth2 / 1000.0  # convert in meters

        if self.is_train:
            # flip = random.randint(0, 1)
            # if flip == 1:
            #     undistorted_raw1, depth_image1 = self.do_flip(undistorted_raw1), self.do_flip(depth_image1)
            #     undistorted_raw2, depth_image2 = self.do_flip(undistorted_raw2), self.do_flip(depth_image2)
            undistorted_raw1, undistorted_depth1 = self.augment_training_data(undistorted_raw1, undistorted_depth1)
            undistorted_raw2, undistorted_depth2 = self.augment_training_data(undistorted_raw2, undistorted_depth2)
        else:
            undistorted_raw1, undistorted_depth1 = self.augment_test_data(undistorted_raw1, undistorted_depth1)
            undistorted_raw2, undistorted_depth2 = self.augment_test_data(undistorted_raw2, undistorted_depth2)

        rot = rel_pose[:-1, :-1].reshape(-1)
        trans = rel_pose[:-1, -1:].reshape(-1)

        angle = np.linalg.norm(w)
        v = w / angle
        v_angle = np.append(v, angle).reshape(-1, 1)
        dt = torch.tensor(dt)

        filename1 = '/'.join(scan_path.split('/')[-2:]).replace('pkl', 'png')
        return {'frame1': undistorted_raw1, 'depth_image1': undistorted_depth1, 'raw_image1': raw_image1,
                'frame2': undistorted_raw2, 'depth_image2': undistorted_depth2, 'raw_image2': raw_image2,
                'filename': filename1, 'imu_data': imu_data, 'dt': dt, 'rel_pose': torch.cat([rot, trans]).float(), 'w': w,'v_angle': v_angle}

    def do_flip(self, img):
        img = cv2.flip(img, 1)  # horizontal flip
        return img

    def get_camera_params(self):
        calib_path = self.cfg.calib_path
        calib = json.load(open(calib_path + '/calibration.json', 'r'))
        cam_fx = calib['camera']['f_x']
        cam_fy = calib['camera']['f_x']
        cam_cx = calib['camera']['c_x']
        cam_cy = calib['camera']['c_y']
        k0 = calib['camera']['k_0']
        k1 = calib['camera']['k_1']
        k2 = calib['camera']['k_2']
        p1 = calib['camera']['p_x']
        p2 = calib['camera']['p_y']

        self.cameraMatrix = np.array([[cam_fx, 0, cam_cx],
                                      [0, cam_fy, cam_cy],
                                      [0, 0, 1]], dtype=np.float64)
        self.distCoeffs = np.array([k0, k1, p1, p2, k2])
