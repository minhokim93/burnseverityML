'''
Deep learning based burn severity prediction using remote sensing data
: PyTorch Data Loader
- Author: Minho Kim (2023)
'''

import torch
from torch.utils.data import Dataset
import random

# Dataset for Dataloaders
class tensorDataset(Dataset):
    
    def __init__(self, images, masks, augmentations=None, seed=42):
        self.images = images
        self.masks  = masks
        self.augmentations = augmentations
        self.seed = seed

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if self.augmentations is not None:
            # Apply augmentations to both image and mask
            random.seed(self.seed)

            image = self.augmentations(image)
            mask = self.augmentations(mask.unsqueeze(0))

        # Turn on gradient for image
        # img = image.detach().clone().requires_grad_(True)
        # mask = mask.long()
        
        return image, mask.long()
    
    def __len__(self):
        return len(self.images)