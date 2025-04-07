import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim

from torch.utils.data import Dataset, DataLoader

import albumentations as A
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
import nibabel as nib
import cv2
import SimpleITK as sitk

transform = transforms.Compose([
    transforms.ToTensor()
])




class transformA(object):
    def __init__(self):
        pass

    def __call__(self, img):
        image = np.array(img).astype(np.float32)
        aug2 = A.Compose([
            A.OneOf([
                # A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.3,
                #                      val_shift_limit=0.3, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.1, p=0.5),
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=(0.0, 0.2)),
            ], p=0.3),
        ])
        augmented2 = aug2(image=image)
        image = augmented2['image'].astype(np.float64)
        return image



class Mydataset(Dataset):
    def __init__(self, root_path, transform=None, dataset='train'):
        self.root_path = root_path
        self.data = dataset
        if transform:
            self.transform = transform()
        else:
            self.transform = transform
        self.origin_path = os.path.join(self.root_path, 'image')
        self.mask_path = os.path.join(self.root_path, 'label')
        self.img_list = sorted(os.listdir(self.origin_path))
    def __getitem__(self, index):
        img_dcm_path = os.path.join(self.origin_path, self.img_list[index])
        mask_path = os.path.join(self.mask_path, self.img_list[index])
        image = sitk.ReadImage(img_dcm_path)
        image_3d = np.squeeze(sitk.GetArrayFromImage(image))
        mask = sitk.ReadImage(mask_path)
        mask = np.squeeze(sitk.GetArrayFromImage(mask))
        mask = np.flipud(mask)
        mask = np.rot90(mask, k=-1)
        image_3d = (image_3d - np.min(image_3d)) / (np.max(image_3d) - np.min(image_3d))
        mask = mask / 255.
        image_out_3d = np.zeros((304, 128, 304))
        for xn in range(image_3d.shape[0]):

            image2 = image_3d[xn, :, :]
            if self.transform:
                image2 = self.transform(image2)
                image_out_3d[xn, :, :] = image2
            else:
                image_out_3d[xn, :, :] = image2

        image_3d, mask = transform(image_3d), transform(mask)
        image_3d = image_3d.unsqueeze(0)
        return image_3d, mask


    def __len__(self):
        return len(self.img_list)
