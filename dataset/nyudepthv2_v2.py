
import os
import cv2
import random

from dataset.base_dataset_v2 import BaseDataset_v2


class nyudepthv2_v2(BaseDataset_v2):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'nyudepthv2')
        if is_train:
            txt_path += '/train_list.txt'
        else:
            txt_path += '/train_list.txt'

        self.dictInTXT = self.readTXT_custom(txt_path, is_train)
        self.make_data_pair()

        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.datadict['gt1'])))

    def __len__(self):
        return len(self.datadict['gt1'])

    def __getitem__(self, idx):
        frame1_path = self.data_path + self.datadict['frame1'][idx]
        frame2_path = self.data_path + self.datadict['frame2'][idx]
        gt1_path = self.data_path + self.datadict['gt1'][idx]
        gt2_path = self.data_path + self.datadict['gt2'][idx]

        frame1 = cv2.imread(frame1_path)                                        # [480, 640, 3]
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        depth1 = cv2.imread(gt1_path, cv2.IMREAD_UNCHANGED).astype('float32')   # [480, 640]

        frame2 = cv2.imread(frame2_path)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        depth2 = cv2.imread(gt2_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.is_train:
            flip = random.randint(0, 1)
            if flip == 1:
                frame1, depth1 = self.do_flip(frame1), self.do_flip(depth1)
                frame2, depth2 = self.do_flip(frame2), self.do_flip(depth2)
            frame1, depth1 = self.augment_training_data(frame1, depth1)
            frame2, depth2 = self.augment_training_data(frame2, depth2)
        else:
            frame1, depth1 = self.augment_test_data(frame1, depth1)
            frame2, depth2 = self.augment_test_data(frame2, depth2)

        depth1 = depth1 / 1000.0  # convert in meters
        depth2 = depth2 / 1000.0  # convert in meters

        return {'frame1': frame1, 'depth1': depth1, 'filename1': frame1_path,
                'frame2': frame2, 'depth2': depth2, 'filename2': frame2_path}

    def do_flip(self, img):
        img = cv2.flip(img, 1)  # horizontal flip
        return img

    def make_data_pair(self):
        assert type(self.dictInTXT) == dict
        datadict = {'frame1': [],
                    'frame2': [],
                    'gt1': [],
                    'gt2': []}

        if self.is_train:
            MAX_IDX_DIFF = 8
            for place in self.dictInTXT.keys():
                imgs = sorted(self.dictInTXT[place]['img'])
                gts = sorted(self.dictInTXT[place]['gt'])

                sample_idx = random.sample(range(len(imgs) - 1), k=len(imgs) - 1)

                for frame1_idx in sample_idx:
                    spare_idx = len(imgs) - 1 - frame1_idx
                    if spare_idx >= MAX_IDX_DIFF:
                        idx_diff = random.randint(1, MAX_IDX_DIFF)
                    else:   # 1 <= spare_idx < 8
                        idx_diff = random.randint(1, spare_idx)

                    frame2_idx = frame1_idx + idx_diff
                    datadict['frame1'].append(imgs[frame1_idx])
                    datadict['frame2'].append(imgs[frame2_idx])
                    datadict['gt1'].append(gts[frame1_idx])
                    datadict['gt2'].append(gts[frame2_idx])
        else:
            for place in self.dictInTXT.keys():
                imgs = sorted(self.dictInTXT[place]['img'])
                gts = sorted(self.dictInTXT[place]['gt'])

                sample_idx = list(range(len(imgs) - 1))             # No shuffle

                for frame1_idx in sample_idx:
                    frame2_idx = frame1_idx + 1                     # Consecutive frames
                    datadict['frame1'].append(imgs[frame1_idx])
                    datadict['frame2'].append(imgs[frame2_idx])
                    datadict['gt1'].append(gts[frame1_idx])
                    datadict['gt2'].append(gts[frame2_idx])

        self.datadict = datadict
