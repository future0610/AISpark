from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
from torchvision import transforms as T
import pandas as pd
import rasterio
import time

DIR = os.path.abspath(__file__ + ("\\" + os.pardir) * 2) + "/data/"

# class Spark(Dataset):
#     MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

#     def __init__(self, root = DIR, train = True, transform = None):
#         super(Spark, self).__init__()
#         self.transform = transform
#         self.train = train
#         self.root = root
#         if train:
#             meta = "train_meta.csv"
#         else:
#             meta = "test_meta.csv"
#         root = os.path.abspath(os.path.join(root, meta))
#         self.data_path = pd.read_csv(root)
#         self.data = []
#         self.targets = []

#     def __len__(self):
#         return len(self.data_path)
    
#     def scan(self):
#         start = time.time()
#         img_dir, mask_dir = self.data_path[self.data_path.columns]
#         total_len = len(self.data_path)
#         for index in range(total_len):
#             img_name, mask_name = self.data_path[img_dir], self.data_path[mask_dir]

#             root = os.path.abspath(os.path.join(self.root, f"{img_dir}/{img_name[index]}"))
#             img = rasterio.open(root).read((7,6,2)).transpose((1, 2, 0))
#             img = np.float32(img) / Spark.MAX_PIXEL_VALUE
            
#             # img = img[np.newaxis]
#             if self.train:
#                 root = os.path.abspath(os.path.join(self.root, f"{mask_dir}/{mask_name[index]}"))
#                 mask = rasterio.open(root).read().transpose((1, 2, 0))
#                 mask = np.float32(mask)
#                 # mask = mask[:, :, :, np.newaxis]
#             else:
#                 mask = None

#             # if index == 0:
#                 # self.data = img
#                 # if self.train:
#                 #     self.targets = mask
#             # else:
#             self.data.append(img)
#             # self.data = np.concatenate([self.data, img], axis = 0)
#             if self.train:
#                 self.targets.append(mask)
#                 # self.targets = np.concatenate([self.targets, mask], axis = 0)

#             print(f"\rScanning...[{100 * (index + 1) / total_len:.2f}%]", end = "")
#         print()
#         print(time.time() - start)
#         return img, mask
    
#     def __getitem__(self, index):
#         # start = time.time()
        
#         img = self.data[index]
#         mask = self.targets[index]

#         if self.transform is not None:
#             augmentation = self.transform(image = img, mask = mask)
#             img = augmentation['image']
#             mask = augmentation['mask']
#         # print(time.time() - start)
#         return img, mask

class Spark(Dataset):
    MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

    def __init__(self, root = DIR, train = True, transform = None):
        super(Spark, self).__init__()
        self.transform = transform
        self.train = train
        self.root = root
        if train:
            meta = "train_meta.csv"
        else:
            meta = "test_meta.csv"
        root = os.path.abspath(os.path.join(root, meta))
        self.data = pd.read_csv(root)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        start = time.time()
        # print(index)
        img_dir, mask_dir = self.data[self.data.columns]
        img_name, mask_name = self.data[img_dir], self.data[mask_dir]

        root = os.path.abspath(os.path.join(self.root, f"{img_dir}/{img_name[index]}"))
        img = rasterio.open(root).read((7,6,2)).transpose((1, 2, 0))
        img = np.float32(img) / Spark.MAX_PIXEL_VALUE
        
        root = os.path.abspath(os.path.join(self.root, f"{mask_dir}/{mask_name[index]}"))
        if self.train:
            mask = rasterio.open(root).read().transpose((1, 2, 0))
            mask = np.float32(mask)
        else:
            mask = None

        if self.train and self.transform is not None:
            augmentation = self.transform(image = img, mask = mask)
            img = augmentation['image']
            mask = augmentation['mask']
        elif not self.train and self.transform is not None:
            augmentation = self.transform(image = img, mask = None)
            img = augmentation['image']
        
        return img, mask
