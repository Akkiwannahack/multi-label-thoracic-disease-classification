"""
build_manifest.py — Phase 1: Dataset Preparation

Reads the NIH ChestX-ray14 dataset, creates patient-wise train/val/test
manifests, and computes class-balance statistics.

Usage (from project root):
    python -m data.build_manifest --config configs/base.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# The 14 official thoracic disease classes (order matters — used as column indices)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def parse_labels(finding_str: str) -> list[int]:
    """Convert pipe-separated finding string to 14-dim multi-hot vector.

    'No Finding' maps to the all-zero vector (no 15th class).
    """
    findings = [f.strip() for f in finding_str.split("|")]
    vector = [0] * len(CLASS_NAMES)
    for f in findings:
        if f in CLASS_NAMES:
            vector[CLASS_NAMES.index(f)] = 1
    return vector


def resolve_image_path(image_name: str, root_dir: str, image_dirs: list[str]) -> str | None:
    """Locate an image across images_001..012 sub-directories."""
    for sub in image_dirs:
        candidate = os.path.join(root_dir, sub, image_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_manifests(cfg: dict) -> None:
    ds = cfg["dataset"]
    root_dir = ds["root_dir"]
    output_dir = cfg["paths"]["manifest_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load master CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(root_dir, ds["csv_file"])
    df = pd.read_csv(csv_path)
    print(f"[manifest] Loaded {len(df)} rows from {csv_path}")

    # Parse multi-hot label vectors
    label_cols = [f"label_{i}" for i in range(14)]
    labels = df["Finding Labels"].apply(parse_labels).tolist()
    label_df = pd.DataFrame(labels, columns=label_cols)
    df = pd.concat([df, label_df], axis=1)

    # ------------------------------------------------------------------
    # 2. Resolve full image paths
    # ------------------------------------------------------------------
    df["image_path"] = df["Image Index"].apply(
        lambda name: resolve_image_path(name, root_dir, ds["image_dirs"])
    )
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[manifest] WARNING: {missing} images could not be resolved — dropping them.")
        df = df.dropna(subset=["image_path"])

    # ------------------------------------------------------------------
    # 3. Patient-wise split using official lists
    # ------------------------------------------------------------------
    train_val_list_path = os.path.join(root_dir, ds["train_val_list"])
    test_list_path = os.path.join(root_dir, ds["test_list"])

    with open(train_val_list_path, "r") as f:
        train_val_images = set(line.strip() for line in f if line.strip())
    with open(test_list_path, "r") as f:
        test_images = set(line.strip() for line in f if line.strip())

    df_train_val = df[df["Image Index"].isin(train_val_images)].copy()
    df_test = df[df["Image Index"].isin(test_images)].copy()

    # Further split train_val → train / val BY PATIENT ID (not by image)
    patient_ids = df_train_val["Patient ID"].unique()
    np.random.seed(42)
    np.random.shuffle(patient_ids)

    val_size = int(len(patient_ids) * ds["val_fraction"])
    val_patients = set(patient_ids[:val_size])
    train_patients = set(patient_ids[val_size:])

    df_train = df_train_val[df_train_val["Patient ID"].isin(train_patients)].copy()
    df_val = df_train_val[df_train_val["Patient ID"].isin(val_patients)].copy()

    # ------------------------------------------------------------------
    # 4. Verify zero patient overlap
    # ------------------------------------------------------------------
    train_pids = set(df_train["Patient ID"].unique())
    val_pids = set(df_val["Patient ID"].unique())
    test_pids = set(df_test["Patient ID"].unique())

    assert len(train_pids & val_pids) == 0, "Patient leak between train and val!"
    assert len(train_pids & test_pids) == 0, "Patient leak between train and test!"
    assert len(val_pids & test_pids) == 0, "Patient leak between val and test!"
    print("[manifest] ✓ Zero patient-ID overlap across splits.")

    # ------------------------------------------------------------------
    # 5. Save manifests
    # ------------------------------------------------------------------
    keep_cols = ["image_path", "Patient ID"] + label_cols

    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        split_df = split_df[keep_cols].copy()
        split_df["split"] = split_name
        out_path = os.path.join(output_dir, f"manifest_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"[manifest] Saved {len(split_df)} rows → {out_path}")

    total = len(df_train) + len(df_val) + len(df_test)
    print(f"[manifest] Total rows across splits: {total}")

    # ------------------------------------------------------------------
    # 6. Class-balance report
    # ------------------------------------------------------------------
    class_balance = {}
    for i, name in enumerate(CLASS_NAMES):
        col = f"label_{i}"
        pos_train = int(df_train[col].sum())
        total_train = len(df_train)
        freq = pos_train / total_train if total_train > 0 else 0.0
        class_balance[name] = {
            "positive_count": pos_train,
            "total_count": total_train,
            "positive_frequency": round(freq, 6),
            "inverse_frequency_weight": round(1.0 / freq, 4) if freq > 0 else 0.0,
        }

    balance_path = os.path.join(output_dir, "class_balance.json")
    with open(balance_path, "w") as f:
        json.dump(class_balance, f, indent=2)
    print(f"[manifest] Saved class-balance report → {balance_path}")

    # Pretty-print summary
    print("\n--- Class Balance Summary (training set) ---")
    for name, info in class_balance.items():
        print(f"  {name:>22s}: {info['positive_count']:>6d} / {info['total_count']}  "
              f"({info['positive_frequency']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build dataset manifests for NIH ChestX-ray14")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    build_manifests(cfg)


if __name__ == "__main__":
    main()
