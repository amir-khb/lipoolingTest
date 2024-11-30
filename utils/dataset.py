# utils/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class WHUBuildingDataset(Dataset):
    def __init__(self, root_dir, domain='aerial', split='train', transform=None):
        """
        Args:
            root_dir: Path to data/WHU
            domain: 'aerial' or 'satellite'
            split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.transform = transform

        # Define paths based on domain and split
        if domain == 'aerial':
            self.img_dir = os.path.join(root_dir, domain, split, 'image')
            self.mask_dir = os.path.join(root_dir, domain, split, 'label')
        else:  # satellite domain
            self.img_dir = os.path.join(root_dir, domain, 'image')
            self.mask_dir = os.path.join(root_dir, domain, 'label')

        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Load image and mask (.tif for both)
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        # Convert mask to binary and add channel dimension
        mask = torch.from_numpy(np.array(mask))
        mask = (mask > 0).long()
        mask = mask.unsqueeze(0)  # Add channel dimension [H, W] -> [1, H, W]

        return image, mask