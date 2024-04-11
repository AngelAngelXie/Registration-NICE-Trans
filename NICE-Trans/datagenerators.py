import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NICE_Transeg_Dataset(Dataset):
    def __init__(self, data_path, device=None, mask_path=None, transform=None):
        self.transform = transform
        self.device = device
        self.data_path = data_path
        self.mask_path = mask_path

        # Get the list of image and mask files
        self.images = sorted(os.listdir(data_path))
        if mask_path is not None:
            self.labels = sorted(os.listdir(mask_path))
        else:
            self.labels = [None] * len(self.images)

        print(f"{data_path.split('/')[-1]} file num: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.data_path, self.images[idx])
        img = np.load(img_path)

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        # Load the label if mask_path is provided
        label = None
        if self.mask_path is not None:
            label_path = os.path.join(self.mask_path, self.labels[idx])
            label = np.load(label_path)
            if self.transform:
                label = self.transform(label)

        # # Move data to the device if provided
        # if self.device:
        #     img = img.to(self.device)
        #     if label is not None:
        #         label = label.to(self.device)
        # Move data to the device if provided
        if self.device:
            img = torch.from_numpy(img).to(self.device)
            if label is not None:
                label = torch.from_numpy(label).to(self.device)

        return img, label