"""
dataset.py — Phase 1: PyTorch Dataset + DataLoader factory

Provides `ChestXrayDataset` and utility functions for building weighted
samplers and DataLoaders from the manifests produced by build_manifest.py.
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data.augmentation import get_train_augmentation, get_val_augmentation
from data.enhancement import enhance_image


class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset backed by a manifest CSV.

    Each sample returns:
        image  — (3, H, W) float32 tensor (enhanced + optionally augmented)
        labels — (14,) float32 tensor (multi-hot)
    """

    def __init__(
        self,
        manifest_path: str,
        cfg: dict,
        split: str = "train",
    ):
        if manifest_path.endswith('.parquet'):
            self.df = pd.read_parquet(manifest_path)
        else:
            self.df = pd.read_csv(manifest_path)
        self.cfg = cfg
        self.split = split
        self.label_cols = [f"label_{i}" for i in range(cfg["dataset"]["num_classes"])]

        enh = cfg["enhancement"]
        self.enh_kwargs = dict(
            image_size=enh["image_size"],
            clahe_clip=enh["clahe_clip_limit"],
            clahe_tile=tuple(enh["clahe_tile_grid"]),
            bilateral_d=enh["bilateral_d"],
            bilateral_sc=enh["bilateral_sigma_color"],
            bilateral_ss=enh["bilateral_sigma_space"],
            sharpen_amount=enh["sharpen_amount"],
            imagenet_mean=tuple(enh["imagenet_mean"]),
            imagenet_std=tuple(enh["imagenet_std"]),
        )

        # --- Optional on-disk cache ---
        self.use_cache = cfg["dataset"].get("use_cache", False)
        self._cache_mmap = None
        self._cache_path: str | None = None
        self._cache_shape: tuple[int, ...] | None = None
        self._cache_lookup: dict[str, int] = {}

        if self.use_cache:
            cache_dir = os.path.join(cfg["paths"]["output_dir"], "cache")
            mmap_path = os.path.join(cache_dir, "images.dat")
            index_path = os.path.join(cache_dir, "cache_index.csv")
            if not os.path.isfile(mmap_path) or not os.path.isfile(index_path):
                raise FileNotFoundError(
                    f"Cache files not found in {cache_dir}. "
                    f"Run: python -m data.build_cache --config <cfg>"
                )
            # Build path → row lookup
            idx_df = pd.read_csv(index_path)
            self._cache_lookup = dict(zip(idx_df["image_path"], idx_df["row"]))
            # Store path/shape for lazy open — do NOT create memmap here
            # because Windows spawns DataLoader workers via pickle, and a
            # memmap pickles as a full in-memory copy of the backing file.
            image_size = enh["image_size"]
            n_rows = len(idx_df)
            self._cache_path = mmap_path
            self._cache_shape = (n_rows, image_size, image_size)

        aug_cfg = cfg["augmentation"]
        if split == "train":
            self.augmentation = get_train_augmentation(
                rotation_limit=aug_cfg["rotation_limit"],
                brightness_limit=aug_cfg["brightness_limit"],
                contrast_limit=aug_cfg["contrast_limit"],
                shift_limit=aug_cfg["shift_limit"],
                scale_limit=aug_cfg["scale_limit"],
                horizontal_flip_p=aug_cfg["horizontal_flip_p"],
            )
        else:
            self.augmentation = get_val_augmentation()

    def __getstate__(self):
        """Strip the memmap before pickling so DataLoader workers don't
        serialize the entire backing file."""
        state = self.__dict__.copy()
        state["_cache_mmap"] = None
        return state

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        if self.use_cache and img_path in self._cache_lookup:
            # Lazy-open memmap on first access in this process/worker
            if self._cache_mmap is None:
                self._cache_mmap = np.memmap(
                    self._cache_path, dtype=np.uint8, mode="r",
                    shape=self._cache_shape,
                )
            # Fast path: read pre-enhanced grayscale from memmap
            cache_row = self._cache_lookup[img_path]
            gray = np.array(self._cache_mmap[cache_row])  # (H, W) uint8, copy
            img_enh = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3) uint8
        else:
            # Original path: imread + full enhancement
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            img_enh = enhance_image(img, normalize=False, **self.enh_kwargs)

        # Albumentations augmentation (expects uint8 HWC)
        augmented = self.augmentation(image=img_enh)
        image_tensor = augmented["image"]  # (3, H, W) float32

        # If augmentation pipeline didn't normalize, do it manually
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        # ImageNet z-score normalization (applied after augmentation)
        mean = torch.tensor(self.cfg["enhancement"]["imagenet_mean"]).view(3, 1, 1)
        std = torch.tensor(self.cfg["enhancement"]["imagenet_std"]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Labels
        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image_tensor, labels


def make_weighted_sampler(
    manifest_path: str,
    class_balance_path: str,
    num_classes: int = 14,
    max_weight: float = 10.0,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler using inverse class-frequency weights.

    Each sample's weight = max inverse-frequency weight across its positive labels.
    This up-weights batches containing rare diseases (e.g. Hernia) without
    duplicating patients (avoids leakage risk of naive oversampling).
    """
    if manifest_path.endswith('.parquet'):
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)
    label_cols = [f"label_{i}" for i in range(num_classes)]

    with open(class_balance_path, "r") as f:
        balance = json.load(f)

    class_weights = []
    for i in range(num_classes):
        name = list(balance.keys())[i]
        w = min(balance[name]["inverse_frequency_weight"], max_weight)
        class_weights.append(w)
    class_weights = np.array(class_weights, dtype=np.float32)

    # Per-sample weight = max weight across its positive labels
    labels_matrix = df[label_cols].values.astype(np.float32)   # (N, 14)
    # For samples with no positive label, use weight = 1.0
    sample_weights = np.ones(len(df), dtype=np.float32)
    for i in range(len(df)):
        pos_indices = np.where(labels_matrix[i] > 0)[0]
        if len(pos_indices) > 0:
            sample_weights[i] = class_weights[pos_indices].sum()

    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(df),
        replacement=True,
    )


def build_dataloaders(
    cfg: dict,
    manifest_dir: str | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from manifests + config.

    Returns dict with keys 'train', 'val', 'test'.
    """
    manifest_dir = manifest_dir or cfg["paths"]["manifest_dir"]
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    loaders = {}

    for split in ["train", "val", "test"]:
        parquet_path = os.path.join(manifest_dir, f"manifest_{split}.parquet")
        csv_path = os.path.join(manifest_dir, f"manifest_{split}.csv")
        
        if os.path.isfile(parquet_path):
            manifest_path = parquet_path
        elif os.path.isfile(csv_path):
            manifest_path = csv_path
        else:
            print(f"[dataset] WARNING: Neither {parquet_path} nor {csv_path} found — skipping {split} loader.")
            continue

        ds = ChestXrayDataset(manifest_path, cfg, split=split)

        sampler = None
        shuffle = False
        if split == "train":
            balance_path = os.path.join(manifest_dir, "class_balance.json")
            if os.path.isfile(balance_path):
                sampler = make_weighted_sampler(
                    manifest_path, balance_path,
                    num_classes=cfg["dataset"]["num_classes"],
                    max_weight=cfg["training"]["max_class_weight"],
                )
            else:
                shuffle = True  # fallback if balance file missing

        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders
