# utils/data_loading.py
from torch.utils.data import DataLoader

from utils.dataset import WHUBuildingDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_data_loaders(root_dir, batch_size=8):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Source domain (Aerial) - using official splits
    aerial_train = WHUBuildingDataset(root_dir, domain='aerial', split='train', transform=transform)
    aerial_val = WHUBuildingDataset(root_dir, domain='aerial', split='val', transform=transform)
    aerial_test = WHUBuildingDataset(root_dir, domain='aerial', split='test', transform=transform)

    # Target domain (Satellite) - no official splits
    satellite_dataset = WHUBuildingDataset(root_dir, domain='satellite', split=None, transform=transform)

    # Create dataloaders
    loaders = {
        'source': {
            'train': DataLoader(aerial_train, batch_size=batch_size, shuffle=True),
            'val': DataLoader(aerial_val, batch_size=batch_size),
            'test': DataLoader(aerial_test, batch_size=batch_size)
        },
        'target': {
            'full': DataLoader(satellite_dataset, batch_size=batch_size)
        }
    }

    return loaders