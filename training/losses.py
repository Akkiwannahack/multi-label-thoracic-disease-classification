"""
losses.py — Loss Functions

Multi-label loss functions for handling severe class imbalance
in the NIH ChestX-ray14 dataset. Includes Asymmetric Loss (ASL),
Focal Loss, Weighted BCE with optional label smoothing, and
plain unweighted BCE.

Fix A changes:
  - Separate explicit settings for loss, positive_weighting, class_weighting.
  - Reject unknown loss names.
  - Log the resolved objective and weight vector at startup.
  - pos_weight mode uses PyTorch's native pos_weight argument.
"""

import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Valid loss names (reject unknown ones)
VALID_LOSS_NAMES = ("bce", "asymmetric", "focal", "weighted_bce")


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
    """Weighted binary cross-entropy — scales the ENTIRE class loss
    (positive and negative terms) by a per-class weight.
    This is NOT the same as PyTorch's pos_weight, which only scales
    the positive term.
    """

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


class PositiveWeightedBCELoss(nn.Module):
    """BCE with PyTorch's native pos_weight — only scales the positive term.

    pos_weight[c] > 1 increases recall for class c.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


def load_class_weights(
    class_balance_path: str,
    max_weight: float = 30.0,
    num_classes: int = 14,
    class_names: list[str] | None = None,
) -> torch.Tensor:
    """Load inverse-frequency weights from class_balance.json, clamped to max_weight.

    Fix A: look up by class name in canonical order (if class_names provided).
    """
    with open(class_balance_path, "r") as f:
        balance = json.load(f)

    if class_names is not None:
        weights = []
        for name in class_names:
            if name not in balance:
                raise KeyError(
                    f"Class '{name}' not found in {class_balance_path}. "
                    f"Available: {list(balance.keys())}"
                )
            w = min(balance[name]["inverse_frequency_weight"], max_weight)
            weights.append(w)
    else:
        weights = []
        for name in balance:
            w = min(balance[name]["inverse_frequency_weight"], max_weight)
            weights.append(w)

    assert len(weights) == num_classes, \
        f"Expected {num_classes} class weights, got {len(weights)}"

    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weights(
    class_balance_path: str,
    num_classes: int = 14,
    class_names: list[str] | None = None,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute pos_weight = min(max_w, max(1, sqrt(N_neg / N_pos)))
    from training counts only. These only scale the positive term of BCE.
    """
    with open(class_balance_path, "r") as f:
        balance = json.load(f)

    if class_names is not None:
        names = class_names
    else:
        names = list(balance.keys())

    weights = []
    for name in names:
        info = balance[name]
        n_pos = info["positive_count"]
        n_total = info["total_count"]
        n_neg = n_total - n_pos
        if n_pos > 0:
            raw = (n_neg / n_pos) ** 0.5
            w = min(max_weight, max(1.0, raw))
        else:
            w = max_weight
        weights.append(w)

    assert len(weights) == num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_loss(cfg: dict, class_balance_path: str | None = None) -> nn.Module:
    """Factory function to build the loss from config.

    Fix A: separate loss/class_weighting/positive_weighting/label_smoothing.
    Rejects unknown loss names.
    """
    t = cfg["training"]
    loss_name = t["loss"]
    label_smoothing = t.get("label_smoothing", 0.0)
    class_names = cfg["dataset"].get("class_names", None)
    num_classes = cfg["dataset"]["num_classes"]

    # --- Fix A: reject unknown loss names ---
    if loss_name not in VALID_LOSS_NAMES:
        raise ValueError(
            f"Unknown loss '{loss_name}'. Valid: {VALID_LOSS_NAMES}"
        )

    # --- Resolve class weighting ---
    class_weighting = t.get("class_weighting", "auto")
    class_weights = None
    if class_weighting not in ("none", None, "auto"):
        raise ValueError(
            f"Unknown class_weighting '{class_weighting}'. "
            f"Valid: 'none', 'auto', None"
        )
    if class_weighting == "auto" and class_balance_path:
        class_weights = load_class_weights(
            class_balance_path,
            max_weight=t.get("max_class_weight", 30.0),
            num_classes=num_classes,
            class_names=class_names,
        )

    # --- Resolve positive weighting ---
    positive_weighting = t.get("positive_weighting", "none")
    pos_weight = None
    if positive_weighting not in ("none", None, "sqrt_ratio"):
        raise ValueError(
            f"Unknown positive_weighting '{positive_weighting}'. "
            f"Valid: 'none', 'sqrt_ratio', None"
        )
    if positive_weighting == "sqrt_ratio" and class_balance_path:
        pos_weight = compute_pos_weights(
            class_balance_path,
            num_classes=num_classes,
            class_names=class_names,
            max_weight=t.get("max_pos_weight", 5.0),
        )

    # --- Build the loss ---
    if loss_name == "bce":
        # True unweighted BCE with optional pos_weight
        criterion = PositiveWeightedBCELoss(
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Loss: BCE (pos_weight=%s, smoothing=%.2f)",
            "none" if pos_weight is None else pos_weight.tolist(),
            label_smoothing,
        )

    elif loss_name == "asymmetric":
        criterion = AsymmetricLoss(
            gamma_neg=t.get("asl_gamma_neg", 4.0),
            gamma_pos=t.get("asl_gamma_pos", 1.0),
            clip=t.get("asl_clip", 0.05),
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Loss: ASL (gamma_neg=%.1f, gamma_pos=%.1f, clip=%.2f, "
            "class_weights=%s, smoothing=%.2f)",
            t.get("asl_gamma_neg", 4.0),
            t.get("asl_gamma_pos", 1.0),
            t.get("asl_clip", 0.05),
            "none" if class_weights is None else "loaded",
            label_smoothing,
        )

    elif loss_name == "focal":
        criterion = MultiLabelFocalLoss(
            gamma=t.get("focal_gamma", 2.0),
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Loss: Focal (gamma=%.1f, class_weights=%s, smoothing=%.2f)",
            t.get("focal_gamma", 2.0),
            "none" if class_weights is None else "loaded",
            label_smoothing,
        )

    elif loss_name == "weighted_bce":
        criterion = WeightedBCELoss(class_weights=class_weights)
        logger.info(
            "Loss: WeightedBCE (class_weights=%s)",
            "none" if class_weights is None else "loaded",
        )

    else:
        # Should never reach here due to validation above
        raise ValueError(f"Unknown loss '{loss_name}'")

    return criterion
