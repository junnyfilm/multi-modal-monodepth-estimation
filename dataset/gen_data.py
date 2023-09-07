# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2

from base_dataset import BaseDataset
import numpy as np
import pandas
from PIL import Image
import json
import pickle
import torch
import random

# ------------------------------------------------------------------------------
# project_dir
#   |__mim
#       |__configs/
#       |__dataset/gen_data.py   --> Must equal with output path of line 29
#       |__models/
#       |__utils/
#       |__test.py
#       |__train.py
#   |__void-dataset-master
# ------------------------------------------------------------------------------
root_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(root_dir, '../../void-dataset-master'))

#### void dataset load, save
imu_path = os.path.join(data_dir, 'scripts')
release_path = os.path.join(data_dir, 'data/void_release/void_1500/data')
calib_path = os.path.join(data_dir, 'calibration')
data_path = os.path.join(data_dir, 'data/void_release')
output_dir = os.path.abspath(os.path.join(data_dir, '..'))

calib = json.load(open(calib_path + '/calibration.json', 'r'))
t_camera_to_body = calib['alignment']['t_camera_to_body']   # 1*3
w_camera_to_body = calib['alignment']['w_camera_to_body']   # 1*3, Rodrigues
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

# camera calibration param
cam_fx = calib['camera']['f_x']
cam_fy = calib['camera']['f_x']
cam_cx = calib['camera']['c_x']
cam_cy = calib['camera']['c_y']
k0 = calib['camera']['k_0']
k1 = calib['camera']['k_1']
k2 = calib['camera']['k_2']
p1 = calib['camera']['p_x']
p2 = calib['camera']['p_y']

folders = sorted(os.listdir(release_path))
for folder in folders:
    dataset_kwargs = {'data_path': imu_path, 'void_release_path': release_path}
    dataset_kwargs['imu_csv'] = os.path.join(imu_path, folder + '-raw-camera-imu.csv')
    dataset_kwargs['image_raw'] = os.path.join(release_path, folder, 'image')
    dataset_kwargs['aligned_depth'] = os.path.join(release_path, folder, 'ground_truth')
    dataset_kwargs['pose'] = os.path.join(release_path, folder, 'absolute_pose')
    dataset_kwargs['output_save'] = os.path.join(output_dir, 'void_dataset_custom', folder)

    imu_data = pandas.read_csv(dataset_kwargs['imu_csv'])
    imu_data['.header.stamp.nsecs'] = imu_data['.header.stamp.nsecs'].apply(lambda x: round(int(str(x)[:4])))

    # Transform imu data to camera coordinate / imu bias applied
    imu_columns = ['.angular_velocity.x', '.angular_velocity.y', '.angular_velocity.z', '.linear_acceleration.x', '.linear_acceleration.y', '.linear_acceleration.z']
    imu_values = imu_data[imu_columns].values
    gyro_homo = np.hstack((imu_values[:, 0:3] - imu_bias_gyro, np.ones((imu_values.shape[0], 1))))   # n*4
    acc_homo = np.hstack((imu_values[:, 3:6] - imu_bias_acc, np.ones((imu_values.shape[0], 1))))     # n*4
    gyro_transformed_values = gyro_homo.dot(np.transpose(T_camera_to_body))                         # n*4
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

        # raw & depth image read
        raw_image_path_1 = os.path.join(dataset_kwargs['image_raw'], file)
        depth_image_path_1 = os.path.join(dataset_kwargs['aligned_depth'], file) # same name as raw image, preprocessing once!

        raw_image_1 = cv2.imread(raw_image_path_1)  # [480, 640, 3]
        # raw_image_1 = cv2.cvtColor(raw_image_1, cv2.COLOR_BGR2RGB)
        depth_image_1 = cv2.imread(depth_image_path_1, cv2.IMREAD_UNCHANGED).astype('float32')  # [480, 640]

        img_time_1 = float(file.split('.')[0]) + float(file.split('.')[1]) * 0.0001
        imu_time = imu_data['.header.stamp.secs'].values + imu_data['.header.stamp.nsecs'].values * 0.0001

        # pose1 read
        pose1 = open(dataset_kwargs['pose'] + '/' + os.path.splitext(file)[0] + '.txt', 'r').read()
        # pose1_mat = [[float(num_str) for num_str in row.split()] for row in pose1.strip().split('\n')]
        # pose1_rot = [row[:3] for row in pose1_mat]
        # pose1_trans = [[row[3]] for row in pose1_mat]
        pose1_mat = np.fromstring(pose1, sep=' ').reshape(3, 4)
        pose1_transform = np.vstack([pose1_mat, [0, 0, 0, 1]])

        random.seed(123)
        rand_idx = random.randint(5, 10)   # randint(10, 15)
        if img_time_1 is not None and (listup.index(file)-rand_idx) >= 0:
            train_dataset=[]

            # raw & depth image read
            raw_image_path_2 = os.path.join(dataset_kwargs['image_raw'], listup[listup.index(file)-rand_idx])  ## previous image
            depth_image_path_2 = os.path.join(dataset_kwargs['aligned_depth'], listup[listup.index(file)-rand_idx]) # same name as raw image, preprocessing once!

            raw_image_2 = cv2.imread(raw_image_path_2)  # [480, 640, 3]
            # raw_image_2 = cv2.cvtColor(raw_image_2, cv2.COLOR_BGR2RGB)
            depth_image_2 = cv2.imread(depth_image_path_2, cv2.IMREAD_UNCHANGED).astype('float32')  # [480, 640]

            img_time_2 = float(listup[listup.index(file)-rand_idx].split('.')[0]) + float(listup[listup.index(file)-rand_idx].split('.')[1]) * 0.0001
            
            # imu data interval
            imu_data_interval = imu_data[(imu_time >= img_time_2) & (imu_time <= img_time_1)]       # img1-time2 / img2-time1
            imu_interval_tensor = torch.tensor(imu_data_interval[imu_columns].values)

            # dt
            imu_interval_time = imu_data_interval['.header.stamp.secs'].values + \
                                imu_data_interval['.header.stamp.nsecs'].values * 0.0001
            dt_sec = imu_interval_time - img_time_2

            # pose2 read
            pose2 = open(dataset_kwargs['pose'] + '/' + os.path.splitext(listup[listup.index(file)-rand_idx])[0] + '.txt', 'r').read()
            # pose2_mat = [[float(num_str) for num_str in row.split()] for row in pose2.strip().split('\n')]
            # pose2_rot = [row[:3] for row in pose2_mat]
            # pose2_trans = [[row[3]] for row in pose2_mat]
            pose2_mat = np.fromstring(pose2, sep=' ').reshape(3, 4)
            pose2_transform = np.vstack([pose2_mat, [0, 0, 0, 1]])
            relative_pose = np.dot(np.linalg.inv(pose2_transform), pose1_transform)
            rvec, _ = cv2.Rodrigues(relative_pose[:3, :3])
            rvec_tensor = torch.tensor(rvec)
            relative_pose_tensor = torch.tensor(relative_pose)

            # undistortion
            h1, w1 = raw_image_1.shape[:2]
            h2, w2 = raw_image_1.shape[:2]
            imgshape = (w1, h1)

            cameraMatrix = np.array([[cam_fx, 0, cam_cx],
                                     [0, cam_fy, cam_cy],
                                     [0, 0, 1]], dtype=np.float64)
            distCoeffs = np.array([k0, k1, p1, p2, k2])

            distortion_mat, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgshape, 0)
            undistorted_raw1 = cv2.undistort(raw_image_1, cameraMatrix, distCoeffs, None, distortion_mat)
            undistorted_raw2 = cv2.undistort(raw_image_2, cameraMatrix, distCoeffs, None, distortion_mat)

            undistorted_depth1 = cv2.undistort(depth_image_1, cameraMatrix, distCoeffs, None, distortion_mat)
            undistorted_depth2 = cv2.undistort(depth_image_2, cameraMatrix, distCoeffs, None, distortion_mat)

            # train set
            train_dataset.append({'raw_image1': raw_image_1, 'raw_image2': raw_image_2,
                                  'depth_image1': depth_image_1, 'depth_image2': depth_image_2,
                                  'undistorted_raw1': undistorted_raw1, 'undistorted_raw2': undistorted_raw2,
                                  'undistorted_depth1': undistorted_depth1, 'undistorted_depth2': undistorted_depth2,
                                  'rel_pose': relative_pose_tensor, 'imu_data': imu_interval_tensor,
                                  'dt': dt_sec, 'Rodrigues': rvec_tensor})

            # Save train data
            os.makedirs(dataset_kwargs['output_save'], exist_ok=True)
            output_file_path = os.path.join(dataset_kwargs['output_save'], file.split('.')[0] + '.' + file.split('.')[1] + ".pkl")
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(train_dataset, output_file)

        else:
            continue
