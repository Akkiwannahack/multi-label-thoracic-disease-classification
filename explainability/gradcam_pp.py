"""
gradcam_pp.py — Phase 6: Grad-CAM++ Explainability

Generates heatmap overlays showing which image regions drive each disease
prediction. Uses pytorch-grad-cam's GradCAMPlusPlus targeting ResNet50's
last conv block (layer4[-1]).

Also provides Score-CAM as an optional cross-check method.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[gradcam] WARNING: pytorch-grad-cam not installed. "
          "Install with: pip install grad-cam")


def get_target_layer(model: nn.Module) -> list[nn.Module]:
    """Get the target layer for Grad-CAM++ (last ResNet50 conv block).

    The gradient still backpropagates through the full graph correctly
    since layer4's output feeds both the GAP branch and (via Mamba)
    the context branch.
    """
    return [model.backbone.layer4[-1]]


def generate_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device | None = None,
    method: str = "gradcam++",
) -> np.ndarray:
    """Generate a Grad-CAM++ (or Score-CAM) heatmap for a specific class.

    Args:
        model: Trained HybridResNetMamba model
        image: (1, 3, H, W) or (3, H, W) preprocessed input tensor
        target_class: Index of the target disease class (0–13)
        device: Target device
        method: "gradcam++" or "scorecam"

    Returns:
        heatmap: (H, W) float32 array in [0, 1] — the raw attention map
    """
    if not GRADCAM_AVAILABLE:
        raise RuntimeError("pytorch-grad-cam is required. Install with: pip install grad-cam")

    if device is None:
        device = next(model.parameters()).device

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    target_layers = get_target_layer(model)
    targets = [ClassifierOutputTarget(target_class)]

    if method == "scorecam":
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Generate CAM
    grayscale_cam = cam(input_tensor=image, targets=targets)
    heatmap = grayscale_cam[0]  # (H, W)

    return heatmap


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay the heatmap on the original image.

    Args:
        original_image: (H, W, 3) uint8 RGB image
        heatmap: (H, W) float32 in [0, 1]
        alpha: blend factor for the heatmap
        colormap: OpenCV colormap

    Returns:
        overlay: (H, W, 3) uint8 RGB blended image
    """
    # Resize heatmap to match image
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Alpha-blend
    overlay = np.uint8(
        (1 - alpha) * original_image.astype(np.float32)
        + alpha * heatmap_color.astype(np.float32)
    )

    return overlay


def generate_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: int,
    alpha: float = 0.4,
    method: str = "gradcam++",
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: generate heatmap + overlay in one call.

    Args:
        model: Trained model
        image_tensor: (3, H, W) or (1, 3, H, W) preprocessed tensor
        original_image: (H, W, 3) uint8 RGB for visualization
        target_class: Disease index (0–13)
        alpha: Heatmap blend factor
        method: "gradcam++" or "scorecam"
        device: Target device

    Returns:
        (heatmap, overlay) — both as numpy arrays
    """
    heatmap = generate_heatmap(model, image_tensor, target_class, device, method)
    overlay = overlay_heatmap(original_image, heatmap, alpha)
    return heatmap, overlay
