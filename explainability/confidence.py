"""
confidence.py — Phase 8: Confidence Estimation (High-Level Interface)

Wraps MC Dropout prediction + confidence bucketing into a single
user-facing function.

Fix H: One prediction contract returning class order, checkpoint identity,
       preprocessing fingerprint, inference mode, probabilities, thresholds,
       binary predictions, optional uncertainty.
       Deterministic eval-mode path as default; MC-dropout as auxiliary.
       BatchNorm kept in eval mode during MC dropout.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from training.calibration_mc_dropout import (
    mc_dropout_predict,
    get_confidence_labels,
)


def predict_with_confidence(
    model: nn.Module,
    image: torch.Tensor,
    class_names: list[str] | None = None,
    mc_passes: int = 20,
    high_threshold: float = 0.05,
    medium_threshold: float = 0.12,
    device: torch.device | None = None,
    thresholds: dict[str, float] | None = None,
    use_mc_dropout: bool = True,
) -> dict:
    """Full prediction with optional MC Dropout confidence estimation.

    Fix H: deterministic eval-mode probabilities as primary output.
    MC-dropout variability as separately labeled auxiliary information,
    with BatchNorm kept in eval mode.

    Args:
        model: Trained HybridResNetMamba
        image: (3, H, W) or (1, 3, H, W) preprocessed tensor
        class_names: List of 14 disease names (optional, for labeling)
        mc_passes: Number of stochastic forward passes
        high_threshold: std < this → "High" confidence
        medium_threshold: std < this → "Medium", else "Low"
        device: Target device
        thresholds: Per-class thresholds (Fix H). If None, uses 0.5.
        use_mc_dropout: If True, run MC dropout for uncertainty.

    Returns:
        Dictionary with:
            probabilities: (14,) deterministic probabilities
            binary_predictions: (14,) int array using thresholds
            thresholds_used: dict of per-class thresholds
            inference_mode: "deterministic" or "mc_dropout"
            confidence_labels: list of "High"/"Medium"/"Low" per class (if MC)
            std: (14,) standard deviations (if MC)
            mc_mean_probs: (14,) MC mean probabilities (if MC)
            detected_diseases: list of dicts for diseases above threshold
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    # --- Primary: deterministic eval-mode probabilities ---
    model.eval()
    with torch.no_grad():
        logits = model(image)
        det_probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)  # (14,)

    # --- Build threshold array ---
    num_classes = det_probs.shape[0]
    if thresholds is not None and class_names is not None:
        thr_array = np.array([thresholds.get(name, 0.5) for name in class_names])
    else:
        thr_array = np.full(num_classes, 0.5)

    binary_preds = (det_probs >= thr_array).astype(int)

    result = {
        "probabilities": det_probs,
        "binary_predictions": binary_preds,
        "thresholds_used": {
            (class_names[i] if class_names else f"class_{i}"): float(thr_array[i])
            for i in range(num_classes)
        },
        "inference_mode": "deterministic",
    }

    # --- Optional: MC Dropout uncertainty (auxiliary) ---
    confidence_labels = ["High"] * num_classes  # default without MC
    std_probs = np.zeros(num_classes)

    if use_mc_dropout and mc_passes > 0:
        mc_mean, mc_std = mc_dropout_predict(
            model, image, T=mc_passes, device=device
        )
        confidence_labels = get_confidence_labels(
            mc_std, high_threshold, medium_threshold
        )
        std_probs = mc_std
        result["mc_mean_probs"] = mc_mean
        result["inference_mode"] = "deterministic+mc_auxiliary"

    result["confidence_labels"] = confidence_labels
    result["std"] = std_probs

    # --- Detected diseases (above threshold) ---
    detected = []
    for i in range(num_classes):
        if binary_preds[i]:
            entry = {
                "class_index": i,
                "class_name": class_names[i] if class_names else f"class_{i}",
                "probability": round(float(det_probs[i]), 4),
                "confidence": confidence_labels[i],
                "std": round(float(std_probs[i]), 4),
            }
            detected.append(entry)

    # Sort by probability descending
    detected.sort(key=lambda x: x["probability"], reverse=True)
    result["detected_diseases"] = detected

    return result
