import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BaseDataset_v2(Dataset):
    def __init__(self, crop_size):

        basic_transform = [
            # A.HorizontalFlip(),
            # A.RandomCrop(crop_size[0], crop_size[1]),
            A.CenterCrop(crop_size[0], crop_size[1]),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        self.basic_transform = basic_transform
        self.to_tensor = transforms.ToTensor()

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def readTXT_custom(self, txt_path, is_train=True):
        dictInTXT = dict()
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        test_place_list = ['bathroom_0019', 'bathroom_0035', 'bathroom_0054',
                           'bedroom_0020', 'bedroom_0045', 'bedroom_0056a', 'bedroom_0059',
                           'bedroom_0082', 'bedroom_0118', 'bedroom_0125b',
                           'bookstore_0001i', 'classroom_0011', 'dining_room_0007',
                           'dining_room_0024', 'furniture_store_0001c', 'home_office_0011',
                           'kitchen_0011a', 'kitchen_0037', 'kitchen_0059',
                           'living_room_0019', 'living_room_0029', 'living_room_0046b',
                           'living_room_0055', 'living_room_0082', 'office_0012',
                           'office_0024', 'reception_room_0001b',
                           ]

        for i in range(len(listInTXT)):
            img_path = listInTXT[i].split(' ')[0]
            gt_path = listInTXT[i].split(' ')[1]
            place, scene = img_path.split('/')[-2], img_path.split('/')[-1]

            if is_train:
                if place not in test_place_list:
                    if place not in dictInTXT.keys():
                        dictInTXT[place] = {'img': [img_path], 'gt': [gt_path]}
                    else:
                        dictInTXT[place]['img'].append(img_path)
                        dictInTXT[place]['gt'].append(gt_path)
            else:
                if place in test_place_list:
                    if place not in dictInTXT.keys():
                        dictInTXT[place] = {'img': [img_path], 'gt': [gt_path]}
                    else:
                        dictInTXT[place]['img'].append(img_path)
                        dictInTXT[place]['gt'].append(gt_path)

        return dictInTXT

    def augment_training_data(self, image, depth):
        H, W, C = image.shape

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()

        return image, depth

    def augment_test_data(self, image, depth):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()

        return image, depth
