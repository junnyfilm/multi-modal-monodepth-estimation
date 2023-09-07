import argparse
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + "/../")
import json
from glob import glob
import numpy as np
from PIL import Image
import random
import cv2
import subprocess
import pandas
import pickle
from utils.transformation_utils import exp_so3, inv_SE3_T_R, inv_SE3_RT, GetRelPose_tail2tail, log_SO3


parser = argparse.ArgumentParser(description='generate_void_pickle')
parser.add_argument('--data_base_dir', type=str, default='/sungsung/void-dataset/data', help='data base dir')
parser.add_argument('--pickle_output_path', type=str, default='./void_pk_dataset/', help='pickle_output_path')
parser.add_argument('--image_interval_range', type=list, default=[5,5], help='pickle_output_path')
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

def undistortion_image(image, K_intrinsic, k0, k1, p1, p2, k2):
    h, w = image.shape[:2]
    imgshape = (w, h)

    distCoeffs = np.array([k0, k1, p1, p2, k2])

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
    return interval_timestamp, imu_interval

def main():
    opt = parser.parse_args()
    DATA_BASE_DIR = opt.data_base_dir
    PK_OUTPUT_PATH = opt.pickle_output_path
    IMAGE_INTERVAL_RANGE = opt.image_interval_range

    ### load calibration
    calib_path = os.path.join(BASE_DIR, 'calibration/calibration.json')
    with open(calib_path, 'r') as json_file:
        calib = json.load(json_file)
    t_c_i = calib['alignment']['t_camera_to_body']   # 1*3
    w_c_i = calib['alignment']['w_camera_to_body']   # 1*3, Rodrigues
    R_c_i = exp_so3(w_c_i)
    R_i_c = R_c_i.transpose()
    RT_i_c = inv_SE3_T_R(t_c_i, R_c_i)

    # imu bias    
    bias_acc = calib['imu']['b_a']
    bias_gyro = calib['imu']['b_g']

    # camera intrinsic param
    # cam_fx = calib['camera']['f_x']
    # cam_fy = calib['camera']['f_x']
    # cam_cx = calib['camera']['c_x']
    # cam_cy = calib['camera']['c_y']
    k0 = calib['camera']['k_0']
    k1 = calib['camera']['k_1']
    k2 = calib['camera']['k_2']
    p1 = calib['camera']['p_x']
    p2 = calib['camera']['p_y']

    #
    release_path = DATA_BASE_DIR + "/void_release/void_1500/data/"
    raw_path = DATA_BASE_DIR + "/void_raw/"
    
    folders = sorted(os.listdir(release_path))
    for folder  in folders:
        raw_folder_path = raw_path + folder + "/"
        # load list of files_path 
        rosbag_path = raw_folder_path + "/raw.bag"   #f or imu
        imu_csv_path = raw_folder_path + "/raw.csv"
        if not os.path.isfile(imu_csv_path):
            convert_ros_csv(rosbag_path)
            os.remove(rosbag_path)
            os.remove(raw_folder_path+"/dataset")
            os.remove(raw_folder_path+"/dataset_1500")
            os.remove(raw_folder_path+"/dataset_500")

    for folder  in folders:
        release_folder_path = release_path + folder + "/"
        raw_folder_path = raw_path + folder + "/"
        output_folder_path = PK_OUTPUT_PATH + "/" + folder + "/"
        os.makedirs(output_folder_path, exist_ok=True)
        
        # load list of files_path 
        imu_csv_path = raw_folder_path + "/raw.csv"
        imu_data = load_imu_csv(imu_csv_path, bias_acc, bias_gyro, R_c_i)
        
        image_list = glob(release_folder_path + "/image/*.png")
        depth_list = glob(release_folder_path + "/ground_truth/*.png")
        pose_list = glob(release_folder_path + "/absolute_pose/*.txt")
        image_list.sort()
        depth_list.sort()
        pose_list.sort()

        K_intrinsic = np.loadtxt(release_folder_path + "/K.txt", dtype=np.float64)

        num_image = len(image_list)
        for idx_first_image in range(num_image):
            print(f'{folder}:{image_list[idx_first_image][image_list[idx_first_image].rfind("/"):]}:{idx_first_image}/{num_image}', end='\r')
            rand_idx = random.randint(IMAGE_INTERVAL_RANGE[0], IMAGE_INTERVAL_RANGE[1])
            idx_second_image = idx_first_image + rand_idx
            if idx_second_image >= num_image:
                break

            image1_stamp = float(image_list[idx_first_image][image_list[idx_first_image].rfind("/")+1:image_list[idx_first_image].rfind(".png")])
            image1 = cv2.imread(image_list[idx_first_image])
            depth1 = load_depth(depth_list[idx_first_image])
            image1_undistort = undistortion_image(image1, K_intrinsic, k0, k1, p1, p2, k2)
            depth1_undistort = undistortion_image(depth1, K_intrinsic, k0, k1, p1, p2, k2)
            pose1  = load_pose(pose_list[idx_first_image])
            
            image2_stamp = float(image_list[idx_second_image][image_list[idx_second_image].rfind("/")+1:image_list[idx_second_image].rfind(".png")])
            image2 = cv2.imread(image_list[idx_second_image])
            depth2 = load_depth(depth_list[idx_second_image])
            image2_undistort = undistortion_image(image2, K_intrinsic, k0, k1, p1, p2, k2)
            depth2_undistort = undistortion_image(depth2, K_intrinsic, k0, k1, p1, p2, k2)
            pose2  = load_pose(pose_list[idx_second_image])

            relative_pose = GetRelPose_tail2tail(pose1, pose2)
            relative_T = relative_pose[:3,3]
            relative_R = relative_pose[:3,:3]
            relative_w = log_SO3(relative_R)
            relative_axis_angle = np.concatenate((relative_w/np.linalg.norm(relative_w), np.array([np.linalg.norm(relative_w)])), axis=0)

            interval_timestamp, imu_interval = get_imu_interval(imu_data, image1_stamp, image2_stamp)

            train_data = {'timestamp1':image1_stamp,
                        'timestamp2':image2_stamp,
                        # 'image1': image1, 
                        # 'image2': image2,
                        # 'depth1': depth1, 
                        # 'depth2': depth2,
                        'image1_undistort': image1_undistort, 
                        'image2_undistort': image2_undistort,
                        'depth1_undistort': depth1_undistort, 
                        'depth2_undistort': depth2_undistort,
                        'relative_T': relative_T,
                        'relative_R': relative_R,
                        'relative_w': relative_w,
                        'relative_axis_angle': relative_axis_angle,
                        'imu_timestamp': interval_timestamp, 
                        'imu_data': imu_interval, 
                        }

            output_file_path = os.path.join(output_folder_path, str(idx_first_image).zfill(8) + ".pkl")
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(train_data, output_file)
        
if __name__ == '__main__':
    main()
