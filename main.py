import torch
from models.segmentation import LIPSegmentationModel


def main():
    # Create model
    model = LIPSegmentationModel(num_classes=19)
    model.eval()  # Set to evaluation mode

    # Test forward pass
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512)
        segmentation, features = model(x, is_training=False)

    print(f"Segmentation output shape: {segmentation.shape}")
    print(f"Available features: {features.keys()}")

    if 'uncertainty4' in features:
        print(f"Uncertainty shape: {features['uncertainty4'].shape}")


if __name__ == "__main__":
    main()