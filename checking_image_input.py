import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from skimage import io

import os

import matplotlib.pyplot as plt


##Testing area
#datadir = "data_folder/my_data"

#one image
image_tensor = np.float32(io.imread(""))


# rescale by maximum and minimum of the image tensor
minX = image_tensor.min()
maxX = image_tensor.max()
image_tensor=(image_tensor-minX)/(maxX-minX)


#image_tensor = image_tensor.max(axis=0)
image_tensor = cv2.resize(image_tensor, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
image_tensor = np.clip(image_tensor, 0, 1)
plt.imshow(image_tensor)
plt.show()
torch.from_numpy(image_tensor).view(1, 64, 64)



'''
class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""
 
    def __call__(self, sample):
        image_tensor = sample['image_tensor']
 
        # rescale by maximum and minimum of the image tensor
        minX = image_tensor.min()
        maxX = image_tensor.max()
        image_tensor=(image_tensor-minX)/(maxX-minX)
 
        # resize the inputs
        # torch image tensor expected for 3D operations is (N, C, D, H, W)
        image_tensor = image_tensor.max(axis=0)
        image_tensor = cv2.resize(image_tensor, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image_tensor = np.clip(image_tensor, 0, 1)
        return torch.from_numpy(image_tensor).view(1, 64, 64)

class ImageDataset(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize()):
        self.datadir = datadir
        self.mode = mode
        self.images = self.load_images()
        self.transform = transform
        self.threshold = 0.74

    # Utility function to load images from a HDF5 file
    def load_images(self):
       
        images_test=[]
        images_train=[]

        for f, i in zip(os.listdir(os.path.join(self.datadir, "single_cell_images")), range(0, len(os.listdir(os.path.join(self.datadir, "single_cell_images"))))):
            basename = os.path.splitext(f)[0]
            fname = os.path.join(os.path.join(self.datadir, "single_cell_images"), f)
            
            # this will need to be revised - very biased 
            if i<11: 
                try:
                    images_test.append({'name': basename, 'label': basename.split("_",1)[1], 'image_tensor': np.float32(io.imread(fname))})
                except Exception as e:   
                    pass
            else:
                try:
                    images_train.append({'name': basename, 'label': basename.split("_",1)[1], 'image_tensor': np.float32(io.imread(fname))})
                except Exception as e:   
                    pass

        if self.mode == 'train':
            return images_train
        elif self.mode == 'test':
            return images_test
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]

        if self.transform:
            # transform the tensor and the particular z-slice
            image_tensor = self.transform(sample)
            return {'image_tensor': image_tensor, 'name': sample['name'], 'label': sample['label']}
        return sample
'''