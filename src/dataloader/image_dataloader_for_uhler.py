import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from skimage import io

import os

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

class NucleiDataset(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize()):
        self.datadir = datadir
        self.mode = mode
        self.images = self.load_images()
        self.transform = transform
        self.threshold = 0.74

    # Utility function to load images from a HDF5 file
    def load_images(self):
        # load labels
        label_data = pd.read_csv(os.path.join(self.datadir, "nuclear_crops_all_experiments/ratio.csv"))
        label_data_2 = pd.read_csv(os.path.join(self.datadir, "nuclear_crops_all_experiments/protein_ratios_full.csv"))
        # those labels that appear in both ratio and protein
        label_data = label_data.merge(label_data_2, how='inner', on='Label')
        #those in both
        label_dict = {name: (float(ratio), np.abs(int(cl)-2)) for (name, ratio, cl) in zip(list(label_data['Label']), list(label_data['Cor/RPL']), list(label_data['mycl']))}
        #those in protein
        label_dict_2 = {name: np.abs(int(cl)-2) for (name, cl) in zip(list(label_data_2['Label']), list(label_data_2['mycl']))}
        del label_data
        del label_data_2
        
    

        # load images
        images_train = []
        images_test = []

        for f in os.listdir(os.path.join(self.datadir, "nuclear_crops_all_experiments/images")):
            basename = os.path.splitext(f)[0]
            
            fname = os.path.join(os.path.join(self.datadir, "nuclear_crops_all_experiments/images"), f)
            if basename in label_dict.keys():
                images_test.append({'name': basename, 'label': label_dict[basename][0], 'image_tensor': np.float32(io.imread(fname)), 'binary_label': label_dict[basename][1]})
            else:
                try:
                    images_train.append({'name': basename, 'label': -1, 'image_tensor': np.float32(io.imread(fname)), 'binary_label': label_dict_2[basename]})
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
            return {'image_tensor': image_tensor, 'name': sample['name'], 'label': sample['label'], 'binary_label': sample['binary_label']}
        return sample

datadir = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/data"
trainset = NucleiDataset(datadir=datadir, mode='train')

# # retrieve dataloader
# trainset = NucleiDataset(datadir=args.datadir, mode='train')
# testset = NucleiDataset(datadir=args.datadir, mode='test')

# train_loader = DataLoader(trainset, batch_size=args.batch_size, drop_last=False, shuffle=True)
# test_loader = DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False)

# print('Data loaded')