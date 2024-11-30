import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.segmentation import LIPSegmentationModel
from utils.data_loading import get_data_loaders


def train_lip_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LIPSegmentationModel(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Get dataloaders
    data_loaders = get_data_loaders('data/WHU', batch_size=8)

    # Training loop
    num_epochs = 10
    best_val_iou = 0

    for epoch in tqdm(range(num_epochs), desc='Epochs',disable=True):
        # Training
        model.train()
        train_loss = 0

        train_pbar = tqdm(data_loaders['source']['train'],
                          desc=f'Training Epoch {epoch + 1}',
                          leave=True)

        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass with is_training=True
            outputs, _ = model(images, is_training=True)
            loss = criterion(outputs, masks.float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_iou = 0

        val_pbar = tqdm(data_loaders['source']['val'],
                        desc='Validation',
                        leave=True)

        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass with is_training=False to enable uncertainty
                outputs, feature_dict = model(images, is_training=False)
                predictions = (torch.sigmoid(outputs) > 0.5).float()

                # Calculate IoU
                intersection = (predictions * masks).sum()
                union = (predictions + masks).gt(0).sum()
                batch_iou = (intersection / (union + 1e-6)).item()
                val_iou += batch_iou

                val_pbar.set_postfix({'IoU': f'{batch_iou:.4f}'})

        val_iou /= len(data_loaders['source']['val'])

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_val_iou,
                'feature_dict': feature_dict,  # Save feature dictionary for analysis
            }, 'best_lip_model.pth')

        # Print epoch results
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss / len(data_loaders["source"]["train"]):.4f}')
        print(f'Validation IoU: {val_iou:.4f}')
        print(f'Best Validation IoU: {best_val_iou:.4f}')
        print('-' * 50)


if __name__ == '__main__':
    train_lip_model()