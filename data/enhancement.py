"""
enhancement.py — Phase 2: Image Enhancement Pipeline

Shared deterministic preprocessing function used by cache build, uncached loading,
evaluation, demo and report.

Fix G changes:
  - Independent booleans for CLAHE, bilateral and unsharp masking.
  - raw_resize reference path that bypasses all three (grayscale→3ch + normalization).
  - Cache fingerprint (preprocessing params, size, dtype, version).
  - One shared deterministic preprocessing function.
"""

import hashlib
import json

import cv2
import numpy as np

# Version tag — bump when the preprocessing pipeline changes behaviour
_ENHANCEMENT_VERSION = "v2.0"


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a single-channel grayscale image (uint8)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def apply_bilateral_denoise(
    image: np.ndarray,
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """Bilateral filter denoising — preserves edges better than Gaussian."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_unsharp_mask(
    image: np.ndarray,
    amount: float = 1.0,
    ksize: int = 5,
) -> np.ndarray:
    """Unsharp-mask sharpening: sharpened = original + amount * (original - blurred)."""
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    # Work in float to avoid clipping during intermediate calculation
    img_f = image.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    sharpened = img_f + amount * (img_f - blur_f)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Replicate single-channel grayscale to 3-channel RGB (for ImageNet-pretrained models)."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.concatenate([image, image, image], axis=-1)
    return image  # already 3-channel


def normalize_imagenet(
    image: np.ndarray,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Histogram-normalize to [0,1], then z-score with ImageNet stats. Returns float32 HWC."""
    img = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean) / std


def enhance_image(
    img: np.ndarray,
    image_size: int = 224,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    bilateral_d: int = 5,
    bilateral_sc: float = 50.0,
    bilateral_ss: float = 50.0,
    sharpen_amount: float = 1.0,
    imagenet_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, ...] = (0.229, 0.224, 0.225),
    normalize: bool = True,
    # --- Fix G: independent booleans ---
    clahe_enabled: bool = True,
    bilateral_enabled: bool = True,
    unsharp_enabled: bool = True,
) -> np.ndarray:
    """Shared deterministic preprocessing pipeline.

    Fix G: independent booleans for CLAHE, bilateral and unsharp.
    When all three are False, this is the 'raw_resize' path.

    Args:
        img: Input image (grayscale uint8, any size).
        normalize: If True, return float32 z-scored; if False, return uint8 RGB
                   (useful for visualization / augmentations that expect uint8).
        clahe_enabled: Apply CLAHE contrast enhancement.
        bilateral_enabled: Apply bilateral denoising.
        unsharp_enabled: Apply unsharp mask sharpening.

    Returns:
        Enhanced image — either float32 HWC (if normalize=True) or uint8 HWC.
    """
    # Ensure grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img

    # 1. Resize (before enhancement so kernel sizes behave consistently)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # 2. CLAHE (optional)
    if clahe_enabled:
        gray = apply_clahe(gray, clip_limit=clahe_clip, tile_grid=clahe_tile)

    # 3. Bilateral denoising (optional)
    if bilateral_enabled:
        gray = apply_bilateral_denoise(gray, d=bilateral_d,
                                       sigma_color=bilateral_sc, sigma_space=bilateral_ss)

    # 4. Unsharp-mask sharpening (optional)
    if unsharp_enabled:
        gray = apply_unsharp_mask(gray, amount=sharpen_amount)

    # 5. Convert to 3-channel RGB
    rgb = grayscale_to_rgb(gray)

    if normalize:
        # 6. Z-score normalization with ImageNet stats
        return normalize_imagenet(rgb, mean=imagenet_mean, std=imagenet_std)
    else:
        return rgb


def get_enhancement_kwargs(cfg: dict) -> dict:
    """Build enhancement kwargs from config, including Fix G booleans."""
    enh = cfg["enhancement"]
    return dict(
        image_size=enh["image_size"],
        clahe_clip=enh.get("clahe_clip_limit", 2.0),
        clahe_tile=tuple(enh.get("clahe_tile_grid", [8, 8])),
        bilateral_d=enh.get("bilateral_d", 5),
        bilateral_sc=enh.get("bilateral_sigma_color", 50.0),
        bilateral_ss=enh.get("bilateral_sigma_space", 50.0),
        sharpen_amount=enh.get("sharpen_amount", 1.0),
        imagenet_mean=tuple(enh.get("imagenet_mean", [0.485, 0.456, 0.406])),
        imagenet_std=tuple(enh.get("imagenet_std", [0.229, 0.224, 0.225])),
        clahe_enabled=enh.get("clahe_enabled", True),
        bilateral_enabled=enh.get("bilateral_enabled", True),
        unsharp_enabled=enh.get("unsharp_enabled", True),
    )


def compute_preprocessing_fingerprint(enh_kwargs: dict) -> str:
    """Compute a deterministic hash of the preprocessing parameters.

    Used to invalidate the cache when parameters change.
    """
    # Only include the params that affect pixel values
    relevant = {
        "version": _ENHANCEMENT_VERSION,
        "image_size": enh_kwargs["image_size"],
        "clahe_enabled": enh_kwargs.get("clahe_enabled", True),
        "bilateral_enabled": enh_kwargs.get("bilateral_enabled", True),
        "unsharp_enabled": enh_kwargs.get("unsharp_enabled", True),
    }
    if relevant["clahe_enabled"]:
        relevant["clahe_clip"] = enh_kwargs.get("clahe_clip", 2.0)
        relevant["clahe_tile"] = enh_kwargs.get("clahe_tile", (8, 8))
    if relevant["bilateral_enabled"]:
        relevant["bilateral_d"] = enh_kwargs.get("bilateral_d", 5)
        relevant["bilateral_sc"] = enh_kwargs.get("bilateral_sc", 50.0)
        relevant["bilateral_ss"] = enh_kwargs.get("bilateral_ss", 50.0)
    if relevant["unsharp_enabled"]:
        relevant["sharpen_amount"] = enh_kwargs.get("sharpen_amount", 1.0)

    canonical = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
