"""
hybrid_model.py — Phase 3c: ResNet50 + Vision Mamba Hybrid

Fuses local features (ResNet50 GAP) with global context (Vision Mamba)
and passes the concatenated vector through a multi-label classification head.

Fix B: explicit parameter groups with freeze_backbone / unfreeze_layer4,
       frozen BN running stats.
Fix D: explicit mamba_backend, residual_fusion, checkpoint provenance.
Fix I: resolution guard (only 224px supported).
"""

import logging

import torch
import torch.nn as nn

from models.resnet_backbone import ResNet50Backbone
from models.vision_mamba_block import (
    VisionMambaEncoder,
    resolve_backend,
    VALID_FUSION_MODES,
    VALID_BACKENDS,
)

logger = logging.getLogger(__name__)


class HybridResNetMamba(nn.Module):
    """Hybrid backbone for multi-label thoracic disease classification.

    Architecture:
        Input (B, 3, 224, 224)
            │
            ▼
        ResNet50 → feature map (B, 2048, 7, 7)
            │                         │
            ▼                         ▼
        GAP → (B, 2048)       VisionMamba → (B, d_model)
            │                         │
            └─── concatenate ─────────┘
                     │
                     ▼
              (B, fused_dim) → LayerNorm → Linear → GELU
                          → Dropout → Linear → GELU
                          → Dropout → Linear → logits
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
        mamba_backend: str | None = None,
        residual_fusion: str = "mean_v2",
    ):
        super().__init__()

        # --- Fix D: explicit backend ---
        if mamba_backend is not None:
            use_cuda = resolve_backend(mamba_backend)
        elif mamba_use_cuda is not None:
            use_cuda = mamba_use_cuda
        else:
            use_cuda = None  # will auto-detect in VisionMambaEncoder

        # Store provenance metadata
        self._mamba_backend = mamba_backend or ("mamba_ssm" if use_cuda else "custom_pytorch")
        self._residual_fusion = residual_fusion
        self._mamba_d_model = mamba_d_model
        self._mamba_num_blocks = mamba_num_blocks
        self._num_classes = num_classes

        # --- ResNet50 backbone ---
        self.backbone = ResNet50Backbone(pretrained=resnet_pretrained)

        # --- Vision Mamba encoder ---
        self.mamba_encoder = VisionMambaEncoder(
            in_channels=2048,
            spatial_size=7,
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            num_blocks=mamba_num_blocks,
            use_cuda=use_cuda,
            residual_fusion=residual_fusion,
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
        mamba_vec = self.mamba_encoder(feat_map)  # (B, d_model)

        # Concatenate and classify
        fused = torch.cat([gap_vec, mamba_vec], dim=1)  # (B, fused_dim)
        logits = self.classifier(fused)  # (B, num_classes)

        return logits

    # ----------------------------------------------------------------
    # Checkpoint provenance (Fix D)
    # ----------------------------------------------------------------

    def get_provenance(self) -> dict:
        """Return metadata for embedding in checkpoints."""
        return {
            "model_type": "hybrid_resnet_mamba",
            "mamba_backend": self._mamba_backend,
            "residual_fusion": self._residual_fusion,
            "mamba_d_model": self._mamba_d_model,
            "mamba_num_blocks": self._mamba_num_blocks,
            "num_classes": self._num_classes,
            "backbone_weights": self.backbone.weight_identifier,
        }

    # ----------------------------------------------------------------
    # Freezing helpers (Fix B)
    # ----------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze ALL backbone parameters (stage-1: only Mamba + head train)."""
        self.backbone.freeze_backbone()

    def unfreeze_layer4(self) -> None:
        """Unfreeze ResNet layer4 only (stage-2)."""
        self.backbone.unfreeze_layer4()

    def freeze_bn_stats(self) -> None:
        """Freeze all BatchNorm running statistics in the backbone.
        Call AFTER model.train().
        """
        ResNet50Backbone.freeze_bn_stats(self.backbone)

    # Legacy compatibility
    def freeze_early_layers(self) -> None:
        """Freeze ResNet50 layer1/layer2 (stage-1). LEGACY."""
        self.backbone.freeze_early_layers()

    def unfreeze_all(self) -> None:
        """Unfreeze everything (stage-2). LEGACY."""
        self.backbone.unfreeze_all()

    # ----------------------------------------------------------------
    # Parameter groups (Fix B)
    # ----------------------------------------------------------------

    def get_param_groups_v2(
        self,
        lr_backbone: float,
        lr_head: float,
        fine_tune_layers: list[str] | None = None,
    ) -> list[dict]:
        """Return parameter groups with explicit module assignment.

        Groups:
          - backbone layers (at lr_backbone) — only those in fine_tune_layers
          - mamba_encoder (at lr_head)
          - classifier (at lr_head)
        """
        groups = self.backbone.get_param_groups_v2(
            lr_backbone, lr_head, fine_tune_layers
        )

        mamba_params = [p for p in self.mamba_encoder.parameters() if p.requires_grad]
        if mamba_params:
            groups.append({
                "params": mamba_params,
                "lr": lr_head,
                "name": "mamba_encoder",
            })

        cls_params = [p for p in self.classifier.parameters() if p.requires_grad]
        if cls_params:
            groups.append({
                "params": cls_params,
                "lr": lr_head,
                "name": "classifier",
            })

        return groups

    # Legacy get_param_groups kept for backward compatibility
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

    # Fix D: explicit backend
    mamba_backend = m.get("mamba_backend", None)
    residual_fusion = m.get("residual_fusion", "mean_v2")

    model = HybridResNetMamba(
        num_classes=m["num_classes"],
        resnet_pretrained=True,
        mamba_d_model=m["mamba_d_model"],
        mamba_d_state=m["mamba_d_state"],
        mamba_num_blocks=m["mamba_num_blocks"],
        fusion_hidden=m["fusion_hidden"],
        dropout=m["dropout"],
        mamba_backend=mamba_backend,
        residual_fusion=residual_fusion,
    )

    logger.info(
        "Built model: backend=%s, fusion=%s, d_model=%d, blocks=%d, classes=%d",
        model._mamba_backend, model._residual_fusion,
        model._mamba_d_model, model._mamba_num_blocks, model._num_classes,
    )

    return model
