import torch
import torch.utils.data
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io

import os

class ToTensorNormalize(object):
    """Convert ndarrays in sample dictionary to Tensors."""
 
    def __call__(self, sample):
        image_tensor = sample['image_tensor']
 
        # rescale by maximum and minimum of the image tensor
        minX = image_tensor.min()
        maxX = image_tensor.max()
        image_tensor=(image_tensor-minX)/(maxX-minX)
 
        # resize the inputs
        # torch image tensor expected for 3D operations is (N, C, D, H, W)
        #image_tensor = image_tensor.max(axis=0)
        image_tensor = cv2.resize(image_tensor, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image_tensor = np.clip(image_tensor, 0, 1) #values smaller than 0 become 0, larger than 1 become 1
        return torch.from_numpy(image_tensor).view(1, 64, 64)

class ImageDataset(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize()):
        self.datadir = datadir
        self.mode = mode
        self.images = self.load_images()
        self.transform = transform

    # Utility function to load images from a HDF5 file
    def load_images(self):
       
        images_test=[]
        images_train=[]

        dir_len = len(os.listdir(os.path.join(self.datadir, "single_cell_images")))

        for f, i in zip(os.listdir(os.path.join(self.datadir, "single_cell_images")), range(0, len(os.listdir(os.path.join(self.datadir, "single_cell_images"))))):
            basename = os.path.splitext(f)[0]
            fname = os.path.join(os.path.join(self.datadir, "single_cell_images"), f)
            
            # this will need to be revised to a proper test-train split - very biased  
            if i< dir_len*0.2:
                try:
                    #lavel is just the barcode coordinate e.g. 02_22 etc
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
        # idx could be one or multiple images depending if you call them as a batch
        sample = self.images[idx]

        if self.transform:
            #this doesn't seem to run
            # transform the tensor and the particular z-slice
            image_tensor = self.transform(sample)
            return {'image_tensor': image_tensor, 'name': sample['name'], 'label': sample['label']}
        
        return sample


class RNA_Dataset(Dataset):
    def __init__(self, datadir):
        self.datadir = datadir
        self.rna_data, self.labels = self._load_rna_data()

    def __len__(self):
        return len(self.rna_data)

    # needed to make each barcode a tensor later
    def __getitem__(self, idx):

        rna_sample = self.rna_data[idx]
        barcode = self.labels[idx]
        return {'tensor': torch.from_numpy(rna_sample).float(), 'binary_label': barcode}

    def _load_rna_data(self):
        # this csv has been run through theis lab tutorial - I haven't currently normalised it 
        data = pd.read_csv(os.path.join(self.datadir, "scRNA_filtered.csv"), index_col=0)
        data = data.transpose()
        labels = list(data.index)
        data = data.values
        
        return data, labels
