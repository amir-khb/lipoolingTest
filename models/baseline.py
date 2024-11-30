import torch
import torch.nn as nn
import torchvision.models as models


class BaselineSegmentation(nn.Module):
    def __init__(self, num_classes=1):  # num_classes=1 for binary building segmentation
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)

        # Remove final layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)  # Output will have shape [B, 1, H, W]
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # Extract features
        features = self.backbone(x)

        # Segmentation head
        out = self.seg_head(features)

        # Upsample to input size
        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out