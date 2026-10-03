"""
resnet_backbone.py — Phase 3a: ResNet50 Feature Extractor

Loads ImageNet-pretrained ResNet50 and strips the final avgpool/fc layers
to expose the (B, 2048, 7, 7) feature map from layer4.

Fix B changes:
  - Explicit parameter groups by module (stem / layer1 / layer2 / layer3 / layer4).
  - Frozen-backbone warm-up: freeze everything except head, then unfreeze layer4 only.
  - Keep pretrained BatchNorm running statistics frozen after every model.train() call.
  - Log each group's learning rate, parameter count and sample names.
  - Assert each trainable parameter is in exactly one group.
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone that outputs the layer4 feature map.

    Output shape: (B, 2048, 7, 7) for a 224×224 input.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self._weight_id = "IMAGENET1K_V2"
        else:
            resnet = models.resnet50(weights=None)
            self._weight_id = "none"

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

    @property
    def weight_identifier(self) -> str:
        """Return the exact weight identifier used at init."""
        return self._weight_id

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

    # ----------------------------------------------------------------
    # Freezing helpers (Fix B)
    # ----------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze ALL backbone parameters (stage-1: only head trains)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_layer4(self) -> None:
        """Unfreeze layer4 only (stage-2 fine-tune target)."""
        for param in self.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all layers (legacy compatibility)."""
        for param in self.parameters():
            param.requires_grad = True

    # Legacy compatibility — kept for old callers
    def freeze_early_layers(self) -> None:
        """Freeze layer1 and layer2 (stage-1 of fine-tuning). LEGACY."""
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False

    # ----------------------------------------------------------------
    # BatchNorm freezing (Fix B)
    # ----------------------------------------------------------------

    @staticmethod
    def freeze_bn_stats(module: nn.Module) -> None:
        """Set all BatchNorm layers in *module* to eval mode so their
        running_mean / running_var stay frozen, even when the parent
        module is in train() mode.  Call this AFTER model.train().
        """
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.SyncBatchNorm)):
                m.eval()

    # ----------------------------------------------------------------
    # Parameter-group helpers (Fix B)
    # ----------------------------------------------------------------

    def get_param_groups_v2(
        self,
        lr_backbone: float,
        lr_head: float,
        fine_tune_layers: list[str] | None = None,
    ) -> list[dict]:
        """Return explicit parameter groups for the optimizer.

        If *fine_tune_layers* is given (e.g. ["layer4"]), only those layers
        get lr_backbone; the rest stay frozen (requires_grad=False).

        Otherwise falls back to the old behaviour of splitting
        early (stem+layer1+layer2) vs late (layer3+layer4).

        Returns a list of dicts suitable for torch.optim.AdamW.
        """
        named_modules_map = {
            "stem": [self.conv1, self.bn1],
            "layer1": [self.layer1],
            "layer2": [self.layer2],
            "layer3": [self.layer3],
            "layer4": [self.layer4],
        }

        groups = []
        if fine_tune_layers is not None:
            for layer_name in fine_tune_layers:
                if layer_name not in named_modules_map:
                    raise ValueError(
                        f"Unknown fine_tune_layer '{layer_name}'. "
                        f"Valid: {list(named_modules_map.keys())}"
                    )
                params = []
                for mod in named_modules_map[layer_name]:
                    params.extend(p for p in mod.parameters() if p.requires_grad)
                if params:
                    groups.append({
                        "params": params,
                        "lr": lr_backbone,
                        "name": f"backbone_{layer_name}",
                    })
        else:
            # Legacy split: early / late
            early = []
            for mod in [self.conv1, self.bn1, self.layer1, self.layer2]:
                early.extend(p for p in mod.parameters() if p.requires_grad)
            late = []
            for mod in [self.layer3, self.layer4]:
                late.extend(p for p in mod.parameters() if p.requires_grad)
            if early:
                groups.append({"params": early, "lr": lr_backbone, "name": "backbone_early"})
            if late:
                groups.append({"params": late, "lr": lr_head, "name": "backbone_late"})

        return groups

    # Legacy get_param_groups kept for backward compatibility
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
