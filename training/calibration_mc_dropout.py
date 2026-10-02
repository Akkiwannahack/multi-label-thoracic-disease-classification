"""
calibration_mc_dropout.py — Phase 8: Monte Carlo Dropout Confidence Estimation

Enables dropout at inference time for uncertainty quantification.
Runs T stochastic forward passes and reports mean probability + confidence bucket.
"""

import torch
import torch.nn as nn
import numpy as np


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable MC Dropout: flip Dropout modules to train mode while keeping
    BatchNorm in eval mode.

    This distinction matters — BatchNorm should use running statistics at
    inference (eval mode), but Dropout must be active (train mode) for
    stochastic forward passes.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def disable_mc_dropout(model: nn.Module) -> None:
    """Restore all modules to eval mode."""
    model.eval()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    image: torch.Tensor,
    T: int = 20,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run T stochastic forward passes with MC Dropout.

    Args:
        model: Trained model (will be put into MC Dropout mode)
        image: (1, 3, H, W) or (3, H, W) input tensor
        T: Number of stochastic forward passes
        device: Target device

    Returns:
        mean_probs: (num_classes,) — mean sigmoid probabilities across T passes
        std_probs:  (num_classes,) — std of sigmoid probabilities across T passes
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    # Set model to eval, then enable dropout only
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    for _ in range(T):
        logits = model(image)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    # Restore full eval
    disable_mc_dropout(model)

    all_probs = np.stack(all_probs, axis=0)  # (T, 1, num_classes)
    all_probs = all_probs.squeeze(1)          # (T, num_classes)

    mean_probs = all_probs.mean(axis=0)       # (num_classes,)
    std_probs = all_probs.std(axis=0)          # (num_classes,)

    return mean_probs, std_probs


def get_confidence_labels(
    std_probs: np.ndarray,
    high_threshold: float = 0.05,
    medium_threshold: float = 0.12,
) -> list[str]:
    """Convert per-class std into confidence bucket labels.

    Args:
        std_probs: (num_classes,) standard deviations from MC Dropout
        high_threshold: std < this → "High"
        medium_threshold: std < this → "Medium", else "Low"

    Returns:
        List of confidence labels per class
    """
    labels = []
    for std in std_probs:
        if std < high_threshold:
            labels.append("High")
        elif std < medium_threshold:
            labels.append("Medium")
        else:
            labels.append("Low")
    return labels
