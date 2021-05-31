# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from PIL import Image
from glob import glob
import numpy as np
import scipy.io as sio
from utils import palette
import torch
import os
import cv2

class CocoStuff10k(BaseDataSet):
    def __init__(self, warp_image = True, **kwargs):
        self.warp_image = warp_image
        self.num_classes = 80
        self.palette = palette.COCO_palette
        super(CocoStuff10k, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train', 'test', 'all']:
            file_list = os.path.join(self.root, 'imageLists', self.split + '.txt')
            self.files = [name.rstrip() for name in tuple(open(file_list, "r"))]
        else: raise ValueError(f"Invalid split name {self.split} choose one of [train, test, all]")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, 'images', image_id + '.jpg')
        label_path = os.path.join(self.root, 'annotations', image_id + '.mat')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        label = sio.loadmat(label_path)['S']
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255
        label[label > 80] = 255
        if self.warp_image:
            image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
            label = np.asarray(Image.fromarray(label).resize((513, 513), resample=Image.NEAREST))
        return image, label, image_id

class CocoStuff164k(BaseDataSet):
    def __init__(self, label_map, things_only, **kwargs):
        self.num_classes = 81 if things_only else 182
        self.palette = palette.COCO_palette
        self.label_map = label_map
        self.things_only = things_only
        # print(label_map)
        super(CocoStuff164k, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train2017', 'val2017']:
            file_list = sorted(glob(os.path.join(self.root, 'images', self.split + '/*.jpg')))
            self.files = [os.path.basename(f).split('.')[0] for f in file_list]
        else: raise ValueError(f"Invalid split name {self.split}, either train2017 or val2017")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, 'images', self.split, image_id + '.jpg')
        label_path = os.path.join(self.root, 'annotations', self.split, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.things_only:
            label[label == 255] = 254
            label += 1
            label = np.take(self.label_map, label).astype(np.uint8)
        return image, label, image_id

def get_parent_class(value, dictionary):
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res

class COCO(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, partition = 'CocoStuff164k',
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False,
                    things_only=True):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        id2trainid_objects = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
                      14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
                      23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                      35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
                      44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47,
                      54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56,
                      63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
                      76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
                      86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
        label_map = [0] * 256
        label_map[255] = 255
        for k, v in id2trainid_objects.items():
            label_map[k] = v + 1

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'label_map': label_map,
            'things_only': things_only
        }

        if partition == 'CocoStuff10k': self.dataset = CocoStuff10k(**kwargs)
        elif partition == 'CocoStuff164k': self.dataset = CocoStuff164k(**kwargs)
        else: raise ValueError(f"Please choose either CocoStuff10k / CocoStuff164k")

        super(COCO, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

