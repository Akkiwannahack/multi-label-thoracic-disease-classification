"""
augmentation.py — Data Augmentation

Albumentations-based augmentation pipeline for chest X-rays.
Only applies to the training split.

Fix: Added config-driven toggles for noise, grid distortion,
     and coarse dropout (per plan §5 recipe R).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(
    rotation_limit: int = 10,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    shift_limit: float = 0.08,
    scale_limit: float = 0.10,
    horizontal_flip_p: float = 0.5,
    noise_enabled: bool = True,
    grid_distortion_enabled: bool = True,
    coarse_dropout_enabled: bool = True,
) -> A.Compose:
    """Training-time augmentation applied AFTER enhancement (on uint8 RGB).

    Args:
        noise_enabled: Include GaussNoise.
        grid_distortion_enabled: Include GridDistortion.
        coarse_dropout_enabled: Include CoarseDropout.
    """
    transforms = [
        A.HorizontalFlip(p=horizontal_flip_p),
        A.Rotate(limit=rotation_limit, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=0,
            border_mode=0,
            p=0.5,
        ),
    ]

    if noise_enabled:
        transforms.append(A.GaussNoise(std_range=(0.012, 0.022), p=0.3))

    if grid_distortion_enabled:
        transforms.append(A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2))

    if coarse_dropout_enabled:
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 20),
            hole_width_range=(8, 20),
            fill=0, p=0.3,
        ))

    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_val_augmentation() -> A.Compose:
    """Validation / test — no augmentation, just convert to tensor."""
    return A.Compose([
        ToTensorV2(),
    ])
