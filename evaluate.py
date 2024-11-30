# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader


def evaluate(model, test_loader, device):
    model.eval()
    total_iou = 0
    num_classes = 7

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for each class
            for cls in range(num_classes):
                intersection = ((preds == cls) & (masks == cls)).sum()
                union = ((preds == cls) | (masks == cls)).sum()
                if union > 0:
                    total_iou += intersection.float() / union.float()

    return total_iou / (len(test_loader) * num_classes)