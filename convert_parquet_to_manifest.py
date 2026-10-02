"""
convert_parquet_to_manifest.py — Convert friend's parquet splits into
manifest CSVs compatible with the training pipeline.

Handles:
  1. Remapping file paths from friend's machine to local paths
  2. Renaming columns to match expected format (image_path, label_0..label_13)
  3. Verifying that images actually exist on disk
  4. Regenerating class_balance.json from the new training set

Usage:
    python convert_parquet_to_manifest.py
"""

import json
import os

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
PARQUET_DIR = r"D:\PBL V\thoracic-xray"           # where train/val/test.parquet live
OUTPUT_DIR  = r"D:\PBL V\thoracic-xray\outputs"    # where manifest CSVs go
LOCAL_RAW   = r"D:\PBL V\data\raw"                 # local path to the raw images

# The 14 classes in the correct order (must match the training config)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def fix_filepath(old_path: str) -> str:
    """Remap friend's path to local path.

    Friend's path:  d:/pbl_5thSem/data/raw\\images_001\\images\\00000007_000.png
    Local path:     D:\\PBL V\\data\\raw\\images_001\\images\\00000007_000.png
    """
    # Normalize separators
    p = old_path.replace("\\", "/").replace("//", "/")

    # Find the images_XXX/images/filename.png part
    # Look for 'images_0' as the anchor
    idx = p.find("images_0")
    if idx == -1:
        # Try images_1 for images_010, images_011, images_012
        idx = p.find("images_1")
    if idx == -1:
        print(f"  WARNING: Cannot parse path: {old_path}")
        return old_path

    relative = p[idx:]  # e.g. "images_001/images/00000007_000.png"
    return os.path.join(LOCAL_RAW, relative)


def convert_split(split_name: str) -> pd.DataFrame:
    """Read a parquet split, remap paths and columns."""
    parquet_path = os.path.join(PARQUET_DIR, f"{split_name}.parquet")
    print(f"\n[convert] Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Rows: {len(df)}")

    # Remap file paths
    df["image_path"] = df["filepath"].apply(fix_filepath)

    # Rename disease columns to label_0 .. label_13
    label_cols = []
    for i, name in enumerate(CLASS_NAMES):
        col_name = f"label_{i}"
        df[col_name] = df[name].astype(float)
        label_cols.append(col_name)

    # Add patient ID and split columns
    df["Patient ID"] = df["patient_id"]
    df["split"] = split_name

    # Keep only the columns the training code expects
    keep_cols = ["image_path", "Patient ID"] + label_cols + ["split"]
    df_out = df[keep_cols].copy()

    # Verify images exist
    exists = df_out["image_path"].apply(os.path.isfile)
    missing = (~exists).sum()
    if missing > 0:
        print(f"  WARNING: {missing}/{len(df_out)} images NOT FOUND on disk!")
        print(f"  Example missing: {df_out.loc[~exists, 'image_path'].iloc[0]}")
        # Drop missing
        df_out = df_out[exists].copy()
        print(f"  Kept {len(df_out)} rows after removing missing images.")
    else:
        print(f"  [OK] All {len(df_out)} images verified on disk.")

    return df_out


def compute_class_balance(df_train: pd.DataFrame) -> dict:
    """Compute class-balance stats from the training set."""
    balance = {}
    for i, name in enumerate(CLASS_NAMES):
        col = f"label_{i}"
        pos = int(df_train[col].sum())
        total = len(df_train)
        freq = pos / total if total > 0 else 0.0
        balance[name] = {
            "positive_count": pos,
            "total_count": total,
            "positive_frequency": round(freq, 6),
            "inverse_frequency_weight": round(1.0 / freq, 4) if freq > 0 else 0.0,
        }
    return balance


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_splits = {}
    for split in ["train", "val", "test"]:
        df = convert_split(split)
        csv_path = os.path.join(OUTPUT_DIR, f"manifest_{split}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}")
        all_splits[split] = df

    # Class balance from training set
    balance = compute_class_balance(all_splits["train"])
    balance_path = os.path.join(OUTPUT_DIR, "class_balance.json")
    with open(balance_path, "w") as f:
        json.dump(balance, f, indent=2)
    print(f"\n[convert] Saved class balance -> {balance_path}")

    # Summary
    print("\n" + "=" * 50)
    print("  Conversion Summary")
    print("=" * 50)
    for split, df in all_splits.items():
        print(f"  {split:>5s}: {len(df):,} images")
    total = sum(len(df) for df in all_splits.values())
    print(f"  {'TOTAL':>5s}: {total:,} images")

    print("\n--- Class Balance (training set) ---")
    for name, info in balance.items():
        print(f"  {name:>22s}: {info['positive_count']:>6d} / {info['total_count']}  "
              f"({info['positive_frequency']*100:.2f}%)")

    print("\n[OK] Done! You can now run training with:")
    print("  python -m training.train --config configs/base.yaml")


if __name__ == "__main__":
    main()
