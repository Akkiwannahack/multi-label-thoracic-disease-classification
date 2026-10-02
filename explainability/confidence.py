"""
confidence.py — Phase 8: Confidence Estimation (High-Level Interface)

Wraps MC Dropout prediction + confidence bucketing into a single
user-facing function.
"""

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
) -> dict:
    """Full prediction with MC Dropout confidence estimation.

    Args:
        model: Trained HybridResNetMamba
        image: (3, H, W) or (1, 3, H, W) preprocessed tensor
        class_names: List of 14 disease names (optional, for labeling)
        mc_passes: Number of stochastic forward passes
        high_threshold: std < this → "High" confidence
        medium_threshold: std < this → "Medium", else "Low"
        device: Target device

    Returns:
        Dictionary with:
            probabilities: (14,) mean probabilities
            confidence_labels: list of "High"/"Medium"/"Low" per class
            std: (14,) standard deviations
            detected_diseases: list of dicts for diseases with prob ≥ 0.5
    """
    mean_probs, std_probs = mc_dropout_predict(
        model, image, T=mc_passes, device=device
    )

    confidence_labels = get_confidence_labels(
        std_probs, high_threshold, medium_threshold
    )

    # Detected diseases (prob ≥ 0.5)
    detected = []
    for i in range(len(mean_probs)):
        if mean_probs[i] >= 0.5:
            entry = {
                "class_index": i,
                "class_name": class_names[i] if class_names else f"class_{i}",
                "probability": round(float(mean_probs[i]), 4),
                "confidence": confidence_labels[i],
                "std": round(float(std_probs[i]), 4),
            }
            detected.append(entry)

    # Sort by probability descending
    detected.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "probabilities": mean_probs,
        "std": std_probs,
        "confidence_labels": confidence_labels,
        "detected_diseases": detected,
    }
