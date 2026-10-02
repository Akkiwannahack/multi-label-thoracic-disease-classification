"""
ablation_study.py — Step 4: Genuine Ablation Study

Trains 6 stripped-down variants to measure the individual contribution of
each technique (focal loss, CLAHE, augmentation, threshold optimization).

All variants use:
  - Same train/val/test split (existing manifests, no reshuffling)
  - Same random seed (42)
  - Same architecture (HybridResNetMamba)
  - Same number of epochs / early-stopping rule
  - Same batch size / mixed precision settings

Variants:
  a) True baseline: BCEWithLogitsLoss, no CLAHE, no augmentation, threshold=0.5
  b) Baseline + focal loss only
  c) Baseline + CLAHE only
  d) Baseline + augmentation only
  e) Current full pipeline (focal + CLAHE + augmentation), threshold=0.5
  f) Full pipeline + optimized thresholds (from Step 1)

Usage (from thoracic-xray/):
    D:\\PBL V\\venv\\Scripts\\python.exe -m training.ablation_study --config configs/base.yaml
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
from training.train import train_one_epoch, validate


# ===================== Dataset Variants =====================================

import cv2
import pandas as pd
from data.enhancement import enhance_image
from data.augmentation import get_train_augmentation, get_val_augmentation


class AblationDataset(torch.utils.data.Dataset):
    """Dataset with toggleable CLAHE and augmentation for ablation study."""

    def __init__(self, manifest_path, cfg, split, use_clahe, use_augmentation):
        if manifest_path.endswith('.parquet'):
            self.df = pd.read_parquet(manifest_path)
        else:
            self.df = pd.read_csv(manifest_path)
        self.cfg = cfg
        self.label_cols = [f"label_{i}" for i in range(cfg["dataset"]["num_classes"])]

        enh = cfg["enhancement"]
        self.image_size = enh["image_size"]
        self.imagenet_mean = tuple(enh["imagenet_mean"])
        self.imagenet_std = tuple(enh["imagenet_std"])

        self.use_clahe = use_clahe
        self.enh_kwargs = dict(
            image_size=self.image_size,
            clahe_clip=enh["clahe_clip_limit"],
            clahe_tile=tuple(enh["clahe_tile_grid"]),
            bilateral_d=enh["bilateral_d"],
            bilateral_sc=enh["bilateral_sigma_color"],
            bilateral_ss=enh["bilateral_sigma_space"],
            sharpen_amount=enh["sharpen_amount"],
            imagenet_mean=self.imagenet_mean,
            imagenet_std=self.imagenet_std,
        )

        if use_augmentation and split == "train":
            aug_cfg = cfg["augmentation"]
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        if self.use_clahe:
            # Full enhancement pipeline (CLAHE + denoise + sharpen)
            img_enh = enhance_image(img, normalize=False, **self.enh_kwargs)
        else:
            # No CLAHE: just resize + grayscale->RGB, no enhancement
            img_resized = cv2.resize(
                img, (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA
            )
            # Convert to 3-channel
            img_enh = np.stack([img_resized, img_resized, img_resized], axis=-1)

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


def build_ablation_dataset(manifest_path, cfg, split, use_clahe, use_augmentation):
    """Factory function for AblationDataset."""
    return AblationDataset(manifest_path, cfg, split, use_clahe, use_augmentation)


# ===================== Training a Single Variant ============================

def train_variant(
    variant_name, cfg, use_focal, use_clahe, use_augmentation,
    device, seed=42,
):
    """Train one ablation variant and return test metrics."""
    print(f"\n{'='*70}")
    print(f"  ABLATION: {variant_name}")
    print(f"{'='*70}")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Build model
    model = build_model(cfg).to(device)

    # Build dataloaders
    manifest_dir = cfg["paths"]["manifest_dir"]
    t = cfg["training"]
    bs = t["batch_size"]
    nw = min(t["num_workers"], 2)

    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(manifest_dir, f"manifest_{split}.csv")
        if not os.path.isfile(csv_path):
            print(f"  WARNING: {csv_path} not found")
            continue
        ds = build_ablation_dataset(
            csv_path, cfg, split,
            use_clahe=use_clahe,
            use_augmentation=use_augmentation,
        )
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == "train"),
            num_workers=nw, pin_memory=True,
            drop_last=(split == "train"),
        )

    # Build loss
    if use_focal:
        balance_path = os.path.join(manifest_dir, "class_balance.json")
        criterion = build_loss(
            cfg,
            class_balance_path=balance_path if os.path.isfile(balance_path) else None,
        ).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)

    # Training config
    class_names = cfg["dataset"]["class_names"]
    num_epochs = t["epochs"]
    patience = t["early_stopping_patience"]
    use_amp = t["mixed_precision"] and device.type == "cuda"

    # Stage 1: freeze early layers
    model.freeze_early_layers()
    param_groups = model.get_param_groups(
        lr_backbone=t["lr_backbone_stage1"],
        lr_head=t["lr_head_stage1"],
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    if t["scheduler"] == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t["cosine_T_0"], T_mult=t["cosine_T_mult"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=t["plateau_factor"],
            patience=t["plateau_patience"]
        )

    scaler = GradScaler(device="cuda") if use_amp else None
    best_macro_auc = 0.0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # Stage transition
        if epoch == t["stage1_epochs"] + 1:
            warmup_epochs = t.get("warmup_epochs_stage2", 2)
            model.unfreeze_all()
            param_groups = model.get_param_groups(
                lr_backbone=t["lr_backbone_stage2"],
                lr_head=t["lr_head_stage2"],
            )
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

            # Linear warmup for backbone (Step 3 fix)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            if t["scheduler"] == "cosine_warm_restarts":
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
                    optimizer, mode="max", factor=t["plateau_factor"],
                    patience=t["plateau_patience"]
                )
            scaler = GradScaler(device="cuda") if use_amp else None

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp
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
    variants = {
        "a": ("True baseline (BCE, no CLAHE, no aug)",
              False, False, False),
        "b": ("Baseline + focal loss",
              True, False, False),
        "c": ("Baseline + CLAHE",
              False, True, False),
        "d": ("Baseline + augmentation",
              False, False, True),
        "e": ("Full pipeline (focal+CLAHE+aug) @ 0.5",
              True, True, True),
    }

    if args.variants == "all":
        selected = list(variants.keys()) + ["f"]
    else:
        selected = [v.strip() for v in args.variants.split(",")]

    all_results = []
    variant_e_probs = None      # stored from variant e for reuse in variant f
    variant_e_targets = None    # stored from variant e for reuse in variant f
    start_time = time.time()

    for key in selected:
        if key == "f":
            # Variant f reuses variant e's exact test probabilities with
            # optimized per-class thresholds — no model, no inference.
            if variant_e_probs is None or variant_e_targets is None:
                print("\n[ablation] ERROR: Cannot run variant f — variant e "
                      "was not run in this session. Re-run with --variants "
                      "that includes 'e' (or use 'all').")
                continue

            # Load optimized thresholds
            thr_path = os.path.join(cfg["paths"]["output_dir"], "optimized_thresholds.json")
            if not os.path.isfile(thr_path):
                print(f"\n[ablation] Skipping variant f — {thr_path} not found")
                continue

            with open(thr_path, "r") as f_thr:
                opt_thr = json.load(f_thr)

            print(f"\n{'='*70}")
            print(f"  ABLATION: Full pipeline + optimized thresholds")
            print(f"{'='*70}")
            print(f"  (Re-using variant e's test probabilities — no re-inference)")

            # Apply optimized thresholds to variant e's probabilities
            thr_array = np.array([opt_thr[name]["threshold"] for name in class_names])
            preds_opt = (variant_e_probs >= thr_array[np.newaxis, :]).astype(int)

            per_class_auc = []
            for i in range(len(class_names)):
                try:
                    auc = roc_auc_score(variant_e_targets[:, i], variant_e_probs[:, i])
                except ValueError:
                    auc = float("nan")
                per_class_auc.append(auc)
            valid_aucs = [a for a in per_class_auc if not np.isnan(a)]

            e_result = next(r for r in all_results if r["variant"].startswith("Full pipeline"))
            result_f = {
                "variant": "Full pipeline + optimized thresholds",
                "macro_auc": round(np.mean(valid_aucs), 4) if valid_aucs else 0.0,
                "macro_f1": round(f1_score(variant_e_targets, preds_opt, average="macro", zero_division=0), 4),
                "macro_precision": round(precision_score(variant_e_targets, preds_opt, average="macro", zero_division=0), 4),
                "macro_recall": round(recall_score(variant_e_targets, preds_opt, average="macro", zero_division=0), 4),
                "best_val_auc": e_result.get("best_val_auc", "—"),
            }
            all_results.append(result_f)
            _save_incremental(all_results, cfg["paths"]["output_dir"])
            print(f"\n  [f] Test results: "
                  f"AUC={result_f['macro_auc']:.4f}  "
                  f"F1={result_f['macro_f1']:.4f}  "
                  f"Prec={result_f['macro_precision']:.4f}  "
                  f"Recall={result_f['macro_recall']:.4f}")
            continue

        name, use_focal, use_clahe, use_aug = variants[key]
        result, probs, targets = train_variant(
            name, cfg, use_focal, use_clahe, use_aug, device
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
    print(f"\n  {'Variant':<45s} {'AUC':>7s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s}")
    print(f"  {'-'*75}")
    for r in all_results:
        print(f"  {r['variant']:<45s} {r['macro_auc']:>7.4f} {r['macro_f1']:>7.4f} "
              f"{r['macro_precision']:>7.4f} {r['macro_recall']:>7.4f}")

    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Final save (same helper, one last overwrite)
    _save_incremental(all_results, cfg["paths"]["output_dir"])
    print(f"  Final results saved.")


if __name__ == "__main__":
    main()
