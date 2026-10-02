"""
hybrid_model.py — Phase 3c: ResNet50 + Vision Mamba Hybrid

Fuses local features (ResNet50 GAP) with global context (Vision Mamba)
and passes the concatenated vector through a multi-label classification head.
"""

import torch
import torch.nn as nn

from models.resnet_backbone import ResNet50Backbone
from models.vision_mamba_block import VisionMambaEncoder


class HybridResNetMamba(nn.Module):
    """Hybrid backbone for multi-label thoracic disease classification.

    Architecture:
        Input (B, 3, 224, 224)
            │
            ▼
        ResNet50 → feature map (B, 2048, 7, 7)
            │                         │
            ▼                         ▼
        GAP → (B, 2048)       VisionMamba → (B, 256)
            │                         │
            └─── concatenate ─────────┘
                     │
                     ▼
              (B, 2304) → LayerNorm → Linear(2304, 512) → GELU
                          → Dropout(0.3) → Linear(512, 14) → logits
    """

    def __init__(
        self,
        num_classes: int = 14,
        resnet_pretrained: bool = True,
        mamba_d_model: int = 256,
        mamba_d_state: int = 16,
        mamba_num_blocks: int = 4,
        fusion_hidden: int = 512,
        dropout: float = 0.3,
        mamba_use_cuda: bool | None = None,
    ):
        super().__init__()

        # --- ResNet50 backbone ---
        self.backbone = ResNet50Backbone(pretrained=resnet_pretrained)

        # --- Vision Mamba encoder ---
        self.mamba_encoder = VisionMambaEncoder(
            in_channels=2048,
            spatial_size=7,
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            num_blocks=mamba_num_blocks,
            use_cuda=mamba_use_cuda,
        )

        # --- Global average pooling for ResNet features ---
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- Fusion classification head ---
        fused_dim = 2048 + mamba_d_model  # 2304 by default
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_hidden // 2, num_classes),
            # No sigmoid — BCEWithLogitsLoss handles it
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)

        Returns:
            logits: (B, num_classes) — raw logits, apply sigmoid at inference
        """
        # ResNet50 feature map
        feat_map = self.backbone(x)  # (B, 2048, 7, 7)

        # Branch 1: Global Average Pooling (local texture/edge signal)
        gap_vec = self.gap(feat_map).flatten(1)  # (B, 2048)

        # Branch 2: Vision Mamba (global context)
        mamba_vec = self.mamba_encoder(feat_map)  # (B, 256)

        # Concatenate and classify
        fused = torch.cat([gap_vec, mamba_vec], dim=1)  # (B, 2304)
        logits = self.classifier(fused)  # (B, 14)

        return logits

    # ----- Convenience methods for staged training -----

    def freeze_early_layers(self) -> None:
        """Freeze ResNet50 layer1/layer2 (stage-1)."""
        self.backbone.freeze_early_layers()

    def unfreeze_all(self) -> None:
        """Unfreeze everything (stage-2)."""
        self.backbone.unfreeze_all()

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
    ) -> list[dict]:
        """Parameter groups for differential learning rates.

        Returns 3 groups:
          1. ResNet early layers (layer1/layer2) — lr_backbone
          2. ResNet late layers (layer3/layer4)  — lr_head
          3. Mamba + classifier                  — lr_head
        """
        groups = self.backbone.get_param_groups(lr_backbone, lr_head)

        # Mamba encoder + classifier head
        mamba_classifier_params = list(self.mamba_encoder.parameters()) + \
                                  list(self.classifier.parameters())
        groups.append({"params": mamba_classifier_params, "lr": lr_head})

        return groups


def build_model(cfg: dict) -> HybridResNetMamba:
    """Factory function to build the hybrid model from config."""
    m = cfg["model"]
    return HybridResNetMamba(
        num_classes=m["num_classes"],
        resnet_pretrained=True,
        mamba_d_model=m["mamba_d_model"],
        mamba_d_state=m["mamba_d_state"],
        mamba_num_blocks=m["mamba_num_blocks"],
        fusion_hidden=m["fusion_hidden"],
        dropout=m["dropout"],
    )
