"""
baseline_resnet50.py — Ablation Baseline: Plain ResNet50 (no Vision Mamba)

Used to measure how much the Vision Mamba branch actually contributes
over a standard CNN baseline, for the project's Phase 10 ablation comparison.

Architecture:
    Input (B, 3, 224, 224)
        │
        ▼
    ResNet50 → feature map (B, 2048, 7, 7)
        │
        ▼
    GAP → (B, 2048)
        │
        ▼
    LayerNorm → Linear(2048, 512) → GELU
        → Dropout(0.3) → Linear(512, 14) → logits
"""

import torch
import torch.nn as nn

from models.resnet_backbone import ResNet50Backbone


class ResNet50Baseline(nn.Module):
    """Plain ResNet50 + GAP + classifier head. No Mamba branch."""

    def __init__(
        self,
        num_classes: int = 14,
        resnet_pretrained: bool = True,
        fusion_hidden: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = ResNet50Backbone(pretrained=resnet_pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)

        gap_dim = 2048
        self.classifier = nn.Sequential(
            nn.LayerNorm(gap_dim),
            nn.Linear(gap_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        gap_vec = self.gap(feat_map).flatten(1)
        logits = self.classifier(gap_vec)
        return logits

    def freeze_early_layers(self) -> None:
        self.backbone.freeze_early_layers()

    def unfreeze_all(self) -> None:
        self.backbone.unfreeze_all()

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        groups = self.backbone.get_param_groups(lr_backbone, lr_head)
        groups.append({"params": self.classifier.parameters(), "lr": lr_head})
        return groups


def build_baseline_model(cfg: dict) -> ResNet50Baseline:
    m = cfg["model"]
    return ResNet50Baseline(
        num_classes=m["num_classes"],
        resnet_pretrained=True,
        fusion_hidden=m["fusion_hidden"],
        dropout=m["dropout"],
    )
