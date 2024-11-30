import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.baseline import BaselineSegmentation
from utils.data_loading import get_data_loaders
from tqdm import tqdm


def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineSegmentation().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Get dataloaders
    data_loaders = get_data_loaders('data/WHU', batch_size=8)

    # Training loop
    num_epochs = 10
    best_val_iou = 0

    # Main epoch loop with progress bar
    for epoch in tqdm(range(num_epochs), desc='Epochs', disable=True):
        # Training
        model.train()
        train_loss = 0

        # Progress bar for training batches
        train_pbar = tqdm(data_loaders['source']['train'],
                          desc=f'Training Epoch {epoch + 1}',
                          leave=True)

        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update training progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_iou = 0

        # Progress bar for validation batches
        val_pbar = tqdm(data_loaders['source']['val'],
                        desc='Validation',
                        leave=True)

        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                predictions = (torch.sigmoid(outputs) > 0.5).float()

                # Calculate IoU
                intersection = (predictions * masks).sum()
                union = (predictions + masks).gt(0).sum()
                batch_iou = (intersection / (union + 1e-6)).item()
                val_iou += batch_iou

                # Update validation progress bar
                val_pbar.set_postfix({'IoU': f'{batch_iou:.4f}'})

        val_iou /= len(data_loaders['source']['val'])

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')

        # Print epoch results
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss / len(data_loaders["source"]["train"]):.4f}')
        print(f'Validation IoU: {val_iou:.4f}')
        print(f'Best Validation IoU: {best_val_iou:.4f}')
        print('-' * 50)


if __name__ == '__main__':
    train()