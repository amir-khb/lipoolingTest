import torch
import torch.nn as nn
from .lip_resnet import resnet50
from .uncertainty import UncertaintyResNet
from .adaptation import TestTimeAdapter


class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        return x


class LIPSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Base LIP-ResNet
        base_model = resnet50(pretrained=False)  # Change to False temporarily
        if hasattr(base_model, 'fc'):
            del base_model.fc  # Remove classification head

        # Add uncertainty
        self.backbone = UncertaintyResNet(base_model)

        # Add decoder
        self.decoder = SegmentationDecoder(2048, num_classes)

        # Test-time adaptation
        self.test_adapter = TestTimeAdapter(self)

    def forward(self, x, is_training=True):
        # Get features and uncertainty
        features, feature_dict = self.backbone(x)

        if not is_training and 'uncertainty4' in feature_dict:
            # Apply test-time adaptation only if uncertainty is available
            features = self.test_adapter.adapt(features, feature_dict['uncertainty4'])

        # Decode to segmentation
        segmentation = self.decoder(features)

        # Upsample to input size
        segmentation = nn.functional.interpolate(
            segmentation, size=x.shape[2:], mode='bilinear', align_corners=True
        )

        return segmentation, feature_dict