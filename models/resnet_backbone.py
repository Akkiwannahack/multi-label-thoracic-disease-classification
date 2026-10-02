"""
resnet_backbone.py — Phase 3a: ResNet50 Feature Extractor

Loads ImageNet-pretrained ResNet50 and strips the final avgpool/fc layers
to expose the (B, 2048, 7, 7) feature map from layer4.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone that outputs the layer4 feature map.

    Output shape: (B, 2048, 7, 7) for a 224×224 input.

    Provides staged freezing:
        - freeze_early_layers()  → freezes layer1 + layer2  (for stage-1 training)
        - unfreeze_all()         → unfreezes everything      (for stage-2 training)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)

        # Keep everything up to and including layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Drop avgpool and fc — we handle pooling in the fusion module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the layer4 feature map.

        Args:
            x: (B, 3, 224, 224)

        Returns:
            Feature map (B, 2048, 7, 7)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def freeze_early_layers(self) -> None:
        """Freeze layer1 and layer2 (stage-1 of fine-tuning)."""
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all layers (stage-2 of fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        """Return parameter groups for the optimizer.

        Early layers (layer1/layer2) get `lr_backbone`.
        Later layers (layer3/layer4) get `lr_head` (higher LR).
        """
        early = []
        late = []
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            early.extend(p for p in module.parameters() if p.requires_grad)
        for module in [self.layer3, self.layer4]:
            late.extend(p for p in module.parameters() if p.requires_grad)

        return [
            {"params": early, "lr": lr_backbone},
            {"params": late, "lr": lr_head},
        ]
