import torch
import torch.nn as nn


class TestTimeAdapter:
    def __init__(self, model, momentum=0.9):
        self.model = model
        self.momentum = momentum

    def calculate_entropy(self, features):
        # Add small epsilon to avoid log(0)
        probs = torch.softmax(features, dim=1) + 1e-10
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy

    def update_lip_params(self, features, uncertainty):
        if uncertainty is None:
            return

        entropy = self.calculate_entropy(features)
        importance = 1.0 / uncertainty

        # Update LIP parameters
        for name, module in self.model.named_modules():
            if 'lip' in name.lower():
                with torch.no_grad():
                    if hasattr(module, 'logit') and hasattr(module.logit, 'weight'):
                        weight = module.logit.weight
                        update = importance * entropy
                        weight.data = self.momentum * weight.data + (1 - self.momentum) * update

    def adapt(self, features, uncertainty):
        self.update_lip_params(features, uncertainty)
        return features