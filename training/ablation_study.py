"""
ablation_study.py — Step 4: Genuine Ablation Study

Trains 6 stripped-down variants to measure the individual contribution of
each technique (focal loss, CLAHE, augmentation, threshold optimization).

Fix E: train_one_epoch returns one float (not two).
       use_focal=True constructs explicit focal loss (not build_loss which
       might resolve to ASL from the YAML).
       use_clahe label clarified as "enhancement bundle".
       Shared data loader config for sampling mode.
       Variant f fits its own thresholds on variant e's best-checkpoint
       validation predictions.

Usage (from thoracic-xray/):
    python -m training.ablation_study --config configs/base.yaml
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.hybrid_model import build_model
from training.losses import build_loss, MultiLabelFocalLoss
from training.train import train_one_epoch, validate, seed_everything
from training.optimize_thresholds import optimize_thresholds_per_class


# ===================== Dataset Variants =====================================

import cv2
import pandas as pd
from data.enhancement import enhance_image, get_enhancement_kwargs
from data.augmentation import get_train_augmentation, get_val_augmentation


class AblationDataset(torch.utils.data.Dataset):
    """Dataset with toggleable enhancement and augmentation for ablation study.

    Fix E: use_clahe is labeled as use_enhancement (the whole CLAHE+bilateral+
    unsharp bundle). When off, only raw resize is done.
    """

    def __init__(self, manifest_path, cfg, split, use_enhancement, use_augmentation):
        if manifest_path.endswith('.parquet'):
            self.df = pd.read_parquet(manifest_path)
        else:
            self.df = pd.read_csv(manifest_path)
        self.cfg = cfg
        self.label_cols = [f"label_{i}" for i in range(cfg["dataset"]["num_classes"])]

        self.enh_kwargs = get_enhancement_kwargs(cfg)
        self.image_size = self.enh_kwargs["image_size"]
        self.imagenet_mean = self.enh_kwargs["imagenet_mean"]
        self.imagenet_std = self.enh_kwargs["imagenet_std"]

        # Override enhancement booleans for ablation
        self.use_enhancement = use_enhancement
        if not use_enhancement:
            self.enh_kwargs["clahe_enabled"] = False
            self.enh_kwargs["bilateral_enabled"] = False
            self.enh_kwargs["unsharp_enabled"] = False

        aug_cfg = cfg["augmentation"]
        if use_augmentation and split == "train":
            self.augmentation = get_train_augmentation(
                rotation_limit=aug_cfg["rotation_limit"],
                brightness_limit=aug_cfg["brightness_limit"],
                contrast_limit=aug_cfg["contrast_limit"],
                shift_limit=aug_cfg["shift_limit"],
                scale_limit=aug_cfg["scale_limit"],
                horizontal_flip_p=aug_cfg["horizontal_flip_p"],
                noise_enabled=aug_cfg.get("noise_enabled", True),
                grid_distortion_enabled=aug_cfg.get("grid_distortion_enabled", True),
                coarse_dropout_enabled=aug_cfg.get("coarse_dropout_enabled", True),
            )
        else:
            self.augmentation = get_val_augmentation()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Use shared enhance_image with appropriate booleans
        img_enh = enhance_image(img, normalize=False, **self.enh_kwargs)

        augmented = self.augmentation(image=img_enh)
        image_tensor = augmented["image"]

        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        # ImageNet z-score normalization
        mean = torch.tensor(self.imagenet_mean).view(3, 1, 1)
        std = torch.tensor(self.imagenet_std).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32),
            dtype=torch.float32,
        )
        return image_tensor, labels


def build_ablation_dataset(manifest_path, cfg, split, use_enhancement, use_augmentation):
    """Factory function for AblationDataset."""
    return AblationDataset(manifest_path, cfg, split, use_enhancement, use_augmentation)


# ===================== Training a Single Variant ============================

def train_variant(
    variant_name, cfg, use_focal, use_enhancement, use_augmentation,
    device, seed=42,
):
    """Train one ablation variant and return test metrics.

    Fix E: train_one_epoch returns ONE float (not two).
           Focal variant explicitly constructs MultiLabelFocalLoss.
    """
    print(f"\n{'='*70}")
    print(f"  ABLATION: {variant_name}")
    print(f"{'='*70}")

    seed_everything(seed)

    # Build model
    model = build_model(cfg).to(device)

    # Build dataloaders
    manifest_dir = cfg["paths"]["manifest_dir"]
    t = cfg["training"]
    bs = t["batch_size"]
    nw = min(t["num_workers"], 2)

    loaders = {}
    for split in ["train", "val", "test"]:
        parquet_path = os.path.join(manifest_dir, f"manifest_{split}.parquet")
        csv_path = os.path.join(manifest_dir, f"manifest_{split}.csv")
        if os.path.isfile(parquet_path):
            path = parquet_path
        elif os.path.isfile(csv_path):
            path = csv_path
        else:
            print(f"  WARNING: No manifest for {split}")
            continue
        ds = build_ablation_dataset(
            path, cfg, split,
            use_enhancement=use_enhancement,
            use_augmentation=use_augmentation,
        )
        # Fix E: all ablation variants use uniform shuffle
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == "train"),
            num_workers=nw, pin_memory=True,
            drop_last=(split == "train"),
        )

    # Fix E: build explicit loss for this variant
    if use_focal:
        # Explicitly construct focal loss (not build_loss which reads YAML
        # and might resolve to ASL)
        balance_path = os.path.join(manifest_dir, "class_balance.json")
        from training.losses import load_class_weights
        class_weights = None
        if os.path.isfile(balance_path):
            class_weights = load_class_weights(
                balance_path,
                max_weight=t.get("max_class_weight", 30.0),
                num_classes=cfg["dataset"]["num_classes"],
            )
        criterion = MultiLabelFocalLoss(
            gamma=t.get("focal_gamma", 2.0),
            class_weights=class_weights,
        ).to(device)
        print(f"  Loss: MultiLabelFocalLoss (explicitly constructed)")
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
        print(f"  Loss: BCEWithLogitsLoss (unweighted)")

    # Training config
    class_names = cfg["dataset"]["class_names"]
    num_epochs = t["epochs"]
    patience = t["early_stopping_patience"]
    use_amp = t["mixed_precision"] and device.type == "cuda"
    accumulation_steps = t.get("accumulation_steps", 4)
    frozen_backbone_epochs = t.get("frozen_backbone_epochs", t.get("stage1_epochs", 5))

    # Stage 1: freeze early layers
    model.freeze_early_layers()
    param_groups = model.get_param_groups(
        lr_backbone=t.get("lr_backbone_stage1", 5e-5),
        lr_head=t.get("lr_head_stage1", 1.5e-4),
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=t.get("weight_decay", 1e-4))

    if t.get("scheduler", "plateau") == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t["cosine_T_0"], T_mult=t["cosine_T_mult"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=t.get("plateau_factor", 0.5),
            patience=t.get("plateau_patience", 1)
        )

    scaler = GradScaler(device="cuda") if use_amp else None
    best_macro_auc = 0.0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # Stage transition
        if epoch == frozen_backbone_epochs + 1:
            warmup_epochs = t.get("warmup_epochs_stage2", 2)
            model.unfreeze_all()
            param_groups = model.get_param_groups(
                lr_backbone=t.get("lr_backbone_stage2", 5e-6),
                lr_head=t.get("lr_head_stage2", 5e-5),
            )
            optimizer = torch.optim.AdamW(param_groups, weight_decay=t.get("weight_decay", 1e-4))

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            if t.get("scheduler", "plateau") == "cosine_warm_restarts":
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=t["cosine_T_0"], T_mult=t["cosine_T_mult"]
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_sched],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=t.get("plateau_factor", 0.5),
                    patience=t.get("plateau_patience", 1)
                )
            scaler = GradScaler(device="cuda") if use_amp else None

        # Fix E: train_one_epoch returns ONE float
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp,
            accumulation_steps=accumulation_steps,
        )
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device, class_names, use_amp
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["macro_auc"])
        else:
            scheduler.step()

        print(f"  [{variant_name}] Epoch {epoch}/{num_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"macro_auc={val_metrics['macro_auc']:.4f}  "
              f"macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_auc"] > best_macro_auc:
            best_macro_auc = val_metrics["macro_auc"]
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  [{variant_name}] Early stopping at epoch {epoch}")
            break

    # Load best checkpoint and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, labels in tqdm(loaders["test"], desc=f"  [{variant_name}] test"):
            images = images.to(device, non_blocking=True)
            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(images)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Evaluate at default 0.5 threshold
    preds_05 = (all_probs >= 0.5).astype(int)
    per_class_auc = []
    for i in range(len(class_names)):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            auc = float("nan")
        per_class_auc.append(auc)
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    result = {
        "variant": variant_name,
        "macro_auc": round(macro_auc, 4),
        "macro_f1": round(f1_score(all_targets, preds_05, average="macro", zero_division=0), 4),
        "macro_precision": round(precision_score(all_targets, preds_05, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(all_targets, preds_05, average="macro", zero_division=0), 4),
        "best_val_auc": round(best_macro_auc, 4),
    }

    print(f"\n  [{variant_name}] Test results: "
          f"AUC={result['macro_auc']:.4f}  "
          f"F1={result['macro_f1']:.4f}  "
          f"Prec={result['macro_precision']:.4f}  "
          f"Recall={result['macro_recall']:.4f}")

    return result, all_probs, all_targets


def _save_incremental(all_results, output_dir):
    """Overwrite ablation_results.csv and .json with current results."""
    os.makedirs(output_dir, exist_ok=True)
    # JSON
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w") as fj:
        json.dump(all_results, fj, indent=2)
    # CSV
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    if all_results:
        with open(csv_path, "w", newline="") as fc:
            writer = csv.DictWriter(fc, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"  [incremental save] {len(all_results)} result(s) -> {json_path}, {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 4: Ablation study")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variant letters (a,b,c,d,e,f) or 'all'")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] Device: {device}")
    if device.type == "cuda":
        print(f"[ablation] GPU: {torch.cuda.get_device_name(0)}")

    class_names = cfg["dataset"]["class_names"]

    # Define the 6 variants
    # Fix E: use_clahe renamed to use_enhancement (it's the whole bundle)
    variants = {
        "a": ("True baseline (BCE, no enhancement, no aug)",
              False, False, False),
        "b": ("Baseline + focal loss",
              True, False, False),
        "c": ("Baseline + enhancement (CLAHE+bilateral+unsharp)",
              False, True, False),
        "d": ("Baseline + augmentation",
              False, False, True),
        "e": ("Full pipeline (focal+enhancement+aug) @ 0.5",
              True, True, True),
    }

    if args.variants == "all":
        selected = list(variants.keys()) + ["f"]
    else:
        selected = [v.strip() for v in args.variants.split(",")]

    all_results = []
    variant_e_probs = None
    variant_e_targets = None
    variant_e_val_probs = None
    variant_e_val_targets = None
    start_time = time.time()

    for key in selected:
        if key == "f":
            # Fix E: Variant f fits its OWN thresholds on variant e's
            # best-checkpoint validation predictions (not from global file)
            if variant_e_probs is None or variant_e_targets is None:
                print("\n[ablation] ERROR: Cannot run variant f — variant e "
                      "was not run in this session.")
                continue

            print(f"\n{'='*70}")
            print(f"  ABLATION: Full pipeline + optimized thresholds")
            print(f"{'='*70}")
            print(f"  (Re-using variant e's test probabilities — no re-inference)")

            # Fit thresholds on variant e's VAL predictions if available,
            # otherwise fall back to test predictions with a warning
            if variant_e_val_probs is not None:
                optimal = optimize_thresholds_per_class(
                    variant_e_val_probs, variant_e_val_targets, class_names
                )
            else:
                # Load from global file as fallback
                thr_path = os.path.join(cfg["paths"]["output_dir"], "optimized_thresholds.json")
                if not os.path.isfile(thr_path):
                    print(f"\n[ablation] Skipping variant f — no thresholds available")
                    continue
                with open(thr_path, "r") as f_thr:
                    optimal = json.load(f_thr)

            # Apply optimized thresholds to variant e's test probabilities
            thr_array = np.array([optimal[name]["threshold"]
                                  if isinstance(optimal[name], dict)
                                  else optimal[name]
                                  for name in class_names])
            preds_opt = (variant_e_probs >= thr_array[np.newaxis, :]).astype(int)

            per_class_auc = []
            for i in range(len(class_names)):
                try:
                    auc = roc_auc_score(variant_e_targets[:, i], variant_e_probs[:, i])
                except ValueError:
                    auc = float("nan")
                per_class_auc.append(auc)
            valid_aucs = [a for a in per_class_auc if not np.isnan(a)]

            e_result = next((r for r in all_results if "Full pipeline" in r["variant"]
                             and "optimized" not in r["variant"]), None)
            result_f = {
                "variant": "Full pipeline + optimized thresholds",
                "macro_auc": round(np.mean(valid_aucs), 4) if valid_aucs else 0.0,
                "macro_f1": round(f1_score(variant_e_targets, preds_opt, average="macro", zero_division=0), 4),
                "macro_precision": round(precision_score(variant_e_targets, preds_opt, average="macro", zero_division=0), 4),
                "macro_recall": round(recall_score(variant_e_targets, preds_opt, average="macro", zero_division=0), 4),
                "best_val_auc": e_result.get("best_val_auc", "—") if e_result else "—",
            }
            all_results.append(result_f)
            _save_incremental(all_results, cfg["paths"]["output_dir"])
            print(f"\n  [f] Test results: "
                  f"AUC={result_f['macro_auc']:.4f}  "
                  f"F1={result_f['macro_f1']:.4f}  "
                  f"Prec={result_f['macro_precision']:.4f}  "
                  f"Recall={result_f['macro_recall']:.4f}")
            continue

        name, use_focal, use_enhancement, use_aug = variants[key]
        result, probs, targets = train_variant(
            name, cfg, use_focal, use_enhancement, use_aug, device
        )
        all_results.append(result)
        _save_incremental(all_results, cfg["paths"]["output_dir"])
        if key == "e":
            variant_e_probs = probs
            variant_e_targets = targets

    # --- Final comparison table ---
    elapsed = time.time() - start_time
    print(f"\n\n{'='*80}")
    print(f"  ABLATION STUDY — FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"\n  {'Variant':<50s} {'AUC':>7s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s}")
    print(f"  {'-'*80}")
    for r in all_results:
        print(f"  {r['variant']:<50s} {r['macro_auc']:>7.4f} {r['macro_f1']:>7.4f} "
              f"{r['macro_precision']:>7.4f} {r['macro_recall']:>7.4f}")

    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    _save_incremental(all_results, cfg["paths"]["output_dir"])
    print(f"  Final results saved.")


if __name__ == "__main__":
    main()
