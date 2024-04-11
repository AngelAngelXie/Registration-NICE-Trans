import os, sys
import numpy as np
import scipy.ndimage
import torch
# from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from os import path
import nibabel as nib # for BraTS2020 nii images

def nii_to_array(nii_file_path):
    # Load the .nii file
    nii_image = nib.load(nii_file_path)
    
    # Convert the image to a numpy array
    image_array = nii_image.get_fdata()
    
    return image_array

class NICE_Transeg_Dataset(Dataset):
    def __init__(self, data_path, device, transform=torch.from_numpy):
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []
        # files = glob(path.join(data_path, "*.pkl")) # for IXI
        files = glob(path.join(data_path, "*.nii")) # for BraTS2020
        self.files = files
        print(f"{data_path.split('/')[-1]} file num: {len(files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # for IXI
        # image, label = np.load(self.files[idx], allow_pickle=True) 

        # for BraTS2020, default dimensions (240, 240, 155)
        nii_file_path = self.files[idx]
        nii_image = nib.load(nii_file_path)
        image, label = nii_image.get_fdata()

        return self.transform(image).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
        # return torch.reshape(self.transform(image)[:,:,:144], (144, 192, 160)).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
    
def print_gpu_usage(note=""):
    print(f"{note}: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))




