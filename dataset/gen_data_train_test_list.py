# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os

#### void dataset load, save
root_dir = os.getcwd()      # must return [.../Project dir/mim/dataset]
data_dir = os.path.abspath(os.path.join(root_dir, '../../void-dataset-master'))     # check this line returns [.../Project dir/void-dataset-master]
customdata = os.path.abspath(os.path.join(root_dir, '../../void_dataset_custom'))   # check this line returns [.../Project dir/void_dataset_custom]

imu_path = os.path.join(data_dir, 'scripts')
release_path = os.path.join(data_dir, 'data/void_release/void_1500/data')
calib_path = os.path.join(data_dir, 'calibration')
data_path = os.path.join(data_dir, 'data/void_release/void_1500')
output_dir = os.path.abspath(os.path.join(data_dir, '..'))

with open(os.path.join(data_path, 'train_image.txt'), 'r') as file:
    trainset = file.readlines()

with open(os.path.join(data_path, 'test_image.txt'), 'r') as file:
    testset = file.readlines()

# read void train / test set
trainsets = []
testsets = []
folders = sorted(os.listdir(release_path))
for folder in folders:
    raw_dir = os.path.join('void_1500/data', folder, 'image')
    custom_folder = os.path.join(output_dir, 'void_dataset_custom', folder)
    pickle_files = sorted(os.listdir(custom_folder))
    for pickle_file in pickle_files:
        if os.path.join(raw_dir, pickle_file.replace('.pkl', '.png') + '\n') in trainset:
            trainsets.append(os.path.join(customdata, folder, pickle_file))
        elif os.path.join(raw_dir, pickle_file.replace('.pkl', '.png') + '\n') in testset:
            testsets.append(os.path.join(customdata, folder, pickle_file))
        else:
            print(f"Unknown pickle file named {os.path.join(custom_folder, pickle_file)}")

# make customset 
train_path = os.path.join(customdata, 'train_custom.txt')
test_path = os.path.join(customdata, 'test_custom.txt')
with open(train_path, 'w') as file:
    file.writelines(line + '\n' for line in trainsets)

with open(test_path, 'w') as file:
    file.writelines(line + '\n' for line in testsets)
