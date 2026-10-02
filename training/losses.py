"""
losses.py — Loss Functions

Multi-label loss functions for handling severe class imbalance
in the NIH ChestX-ray14 dataset. Includes Asymmetric Loss (ASL),
Focal Loss, and Weighted BCE with optional label smoothing.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    From 'Asymmetric Loss For Multi-Label Classification' (Ben-Baruch et al., 2021).
    Applies different focusing (gamma) to positive vs negative predictions,
    with optional probability clipping on negatives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1 - probs

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = -loss_pos - loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * (1 - pt) ** gamma

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return loss.mean()


class MultiLabelFocalLoss(nn.Module):
    """Per-class binary focal loss for multi-label classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """Weighted binary cross-entropy as a simpler alternative."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.class_weights is not None:
            bce = bce * self.class_weights.unsqueeze(0)
        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        return bce


def load_class_weights(
    class_balance_path: str,
    max_weight: float = 30.0,
    num_classes: int = 14,
) -> torch.Tensor:
    """Load inverse-frequency weights from class_balance.json, clamped to max_weight."""
    with open(class_balance_path, "r") as f:
        balance = json.load(f)

    weights = []
    for name in balance:
        w = min(balance[name]["inverse_frequency_weight"], max_weight)
        weights.append(w)

    assert len(weights) == num_classes, \
        f"Expected {num_classes} class weights, got {len(weights)}"

    return torch.tensor(weights, dtype=torch.float32)


def build_loss(cfg: dict, class_balance_path: str | None = None) -> nn.Module:
    """Factory function to build the loss from config."""
    t = cfg["training"]
    class_weights = None
    if class_balance_path:
        class_weights = load_class_weights(
            class_balance_path,
            max_weight=t["max_class_weight"],
            num_classes=cfg["dataset"]["num_classes"],
        )

    if t["loss"] == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=t.get("asl_gamma_neg", 4.0),
            gamma_pos=t.get("asl_gamma_pos", 1.0),
            clip=t.get("asl_clip", 0.05),
            class_weights=class_weights,
            label_smoothing=t.get("label_smoothing", 0.0),
        )
    elif t["loss"] == "focal":
        return MultiLabelFocalLoss(
            gamma=t["focal_gamma"],
            class_weights=class_weights,
            label_smoothing=t.get("label_smoothing", 0.0),
        )
    else:
        return WeightedBCELoss(class_weights=class_weights)
