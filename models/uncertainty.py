import torch
import torch.nn as nn


class UncertaintyModule(nn.Module):
    def __init__(self, dropout_rate=0.1, n_samples=5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.n_samples = n_samples

    def forward(self, x):
        if not self.training:
            samples = []
            for _ in range(self.n_samples):
                samples.append(self.dropout(x))

            samples = torch.stack(samples)
            mean = samples.mean(0)
            # Add small epsilon to avoid division by zero
            uncertainty = samples.var(0) + 1e-10
            return mean, uncertainty
        return x, torch.zeros_like(x)


class UncertaintyResNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

        # Add uncertainty modules
        self.uncertainty_modules = nn.ModuleDict({
            'layer1': UncertaintyModule(),
            'layer2': UncertaintyModule(),
            'layer3': UncertaintyModule(),
            'layer4': UncertaintyModule()
        })

        self.features = {}

    def forward(self, x):
        # Initial layers
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        self.features['stem'] = x

        # Layer blocks with uncertainty
        x = self.base.layer1(x)
        x, unc1 = self.uncertainty_modules['layer1'](x)
        self.features['layer1'] = x
        self.features['uncertainty1'] = unc1

        x = self.base.layer2(x)
        x, unc2 = self.uncertainty_modules['layer2'](x)
        self.features['layer2'] = x
        self.features['uncertainty2'] = unc2

        x = self.base.layer3(x)
        x, unc3 = self.uncertainty_modules['layer3'](x)
        self.features['layer3'] = x
        self.features['uncertainty3'] = unc3

        x = self.base.layer4(x)
        x, unc4 = self.uncertainty_modules['layer4'](x)
        self.features['layer4'] = x
        self.features['uncertainty4'] = unc4

        return x, self.features