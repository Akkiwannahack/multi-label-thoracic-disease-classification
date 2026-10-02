"""
verify_cache.py — Verify that the cached path produces identical tensors
to the original imread + enhance path (with augmentation disabled).

Usage:
    python -m data.verify_cache --config configs/base.yaml
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from data.enhancement import enhance_image
from data.augmentation import get_val_augmentation


def verify(cfg: dict, n_samples: int = 20) -> bool:
    enh = cfg["enhancement"]
    image_size = enh["image_size"]
    enh_kwargs = dict(
        image_size=image_size,
        clahe_clip=enh["clahe_clip_limit"],
        clahe_tile=tuple(enh["clahe_tile_grid"]),
        bilateral_d=enh["bilateral_d"],
        bilateral_sc=enh["bilateral_sigma_color"],
        bilateral_ss=enh["bilateral_sigma_space"],
        sharpen_amount=enh["sharpen_amount"],
        imagenet_mean=tuple(enh["imagenet_mean"]),
        imagenet_std=tuple(enh["imagenet_std"]),
    )

    # Load cache
    cache_dir = os.path.join(cfg["paths"]["output_dir"], "cache")
    mmap_path = os.path.join(cache_dir, "images.dat")
    index_path = os.path.join(cache_dir, "cache_index.csv")

    if not os.path.isfile(mmap_path) or not os.path.isfile(index_path):
        print(f"[verify] Cache not found at {cache_dir}. Build it first.")
        return False

    idx_df = pd.read_csv(index_path)
    cache_lookup = dict(zip(idx_df["image_path"], idx_df["row"]))
    n_rows = len(idx_df)
    mmap = np.memmap(mmap_path, dtype=np.uint8, mode="r",
                     shape=(n_rows, image_size, image_size))

    # Pick random training images
    manifest_dir = cfg["paths"]["manifest_dir"]
    train_csv = os.path.join(manifest_dir, "manifest_train.csv")
    df = pd.read_csv(train_csv)
    sample_rows = df.sample(n=min(n_samples, len(df)), random_state=42)

    # Validation augmentation only (no random transforms)
    aug = get_val_augmentation()

    max_diff = 0.0
    all_ok = True

    for i, (_, row) in enumerate(sample_rows.iterrows()):
        img_path = row["image_path"]

        # --- Original path ---
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [{i}] Cannot read {img_path}")
            all_ok = False
            continue
        img_enh_orig = enhance_image(img, normalize=False, **enh_kwargs)
        tensor_orig = aug(image=img_enh_orig)["image"].float() / 255.0
        mean = torch.tensor(enh_kwargs["imagenet_mean"]).view(3, 1, 1)
        std = torch.tensor(enh_kwargs["imagenet_std"]).view(3, 1, 1)
        tensor_orig = (tensor_orig - mean) / std

        # --- Cached path ---
        if img_path not in cache_lookup:
            print(f"  [{i}] {img_path} NOT in cache")
            all_ok = False
            continue
        cache_row = cache_lookup[img_path]
        gray = np.array(mmap[cache_row])
        img_enh_cache = np.stack([gray, gray, gray], axis=-1)
        tensor_cache = aug(image=img_enh_cache)["image"].float() / 255.0
        tensor_cache = (tensor_cache - mean) / std

        diff = (tensor_orig - tensor_cache).abs().max().item()
        max_diff = max(max_diff, diff)

        status = "PASS" if diff == 0.0 else f"FAIL diff={diff:.6f}"
        print(f"  [{i:2d}] {os.path.basename(img_path):30s}  {status}")

        if diff != 0.0:
            all_ok = False

    print(f"\n[verify] Max absolute difference: {max_diff}")
    if all_ok:
        print("[verify] PASSED - cached and original paths are identical.")
    else:
        print("[verify] FAILED - mismatch detected.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ok = verify(cfg, n_samples=args.samples)
    sys.exit(0 if ok else 1)
