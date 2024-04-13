# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class NICE_Transeg_Dataset(Dataset):
#     def __init__(self, data_path, device=None, mask_path=None, transform=None):
#         self.transform = transform
#         self.device = device
#         self.data_path = data_path
#         self.mask_path = mask_path

#         # Get the list of image and mask files
#         self.images = sorted(os.listdir(data_path))
#         if mask_path is not None:
#             self.labels = sorted(os.listdir(mask_path))
#         else:
#             self.labels = [None] * len(self.images)

#         print(f"{data_path.split('/')[-1]} file num: {len(self.images)}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # Load the image
#         img_path = os.path.join(self.data_path, self.images[idx])
#         img = np.load(img_path)

#         # Apply transformations to the image
#         if self.transform:
#             img = self.transform(img)

#         # Load the label if mask_path is provided
#         label = None
#         if self.mask_path is not None:
#             label_path = os.path.join(self.mask_path, self.labels[idx])
#             label = np.load(label_path)
#             if self.transform:
#                 label = self.transform(label)

#         # # Move data to the device if provided
#         # if self.device:
#         #     img = img.to(self.device)
#         #     if label is not None:
#         #         label = label.to(self.device)
#         # Move data to the device if provided
#         if self.device:
#             img = torch.from_numpy(img).permute(3,0,1,2).float().to(self.device)
#             if label is not None:
#                 label = torch.from_numpy(label).permute(3,0,1,2).float().to(self.device)

#         return img, label


import os, sys
import numpy as np
import scipy.ndimage
import torch
# from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from os import path

class NICE_Transeg_Dataset(Dataset):
    def __init__(self, data_path, device, transform=torch.from_numpy):
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []
        files = glob(path.join(data_path, "*.pkl"))
        self.files = files
        print(f"{data_path.split('/')[-1]} file num: {len(files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = np.load(self.files[idx], allow_pickle=True)
        return self.transform(image).unsqueeze(0).to(self.device), self.transform(label).float().unsqueeze(0).to(self.device)
        # return torch.reshape(self.transform(image)[:,:,:144], (144, 192, 160)).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
    
def print_gpu_usage(note=""):
    print(f"{note}: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))