"""
optimize_thresholds.py — Step 1: Per-Class Threshold Optimization

Loads the best checkpoint, runs inference on the VALIDATION set only,
sweeps per-class thresholds to maximize each class's F1, then applies
the locked thresholds to the TEST set for a before/after comparison.

No retraining is performed.

Usage (from thoracic-xray/):
    python -m training.optimize_thresholds --config configs/base.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.hybrid_model import build_model
from data.dataset import ChestXrayDataset






@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference and collect all sigmoid probabilities and targets."""
    model.eval()
    all_probs = []
    all_targets = []

    for images, labels in tqdm(loader, desc="  inference"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_targets)


def optimize_thresholds_per_class(probs, targets, class_names):
    """For each class, sweep thresholds 0.05–0.95 and pick the one that
    maximizes that class's F1 score on the validation data.

    Returns dict: {class_name: {"threshold": float, "val_f1": float, "val_f1_at_0.5": float}}
    """
    thresholds_to_try = np.arange(0.05, 0.96, 0.01)

    optimal = {}

    print(f"\n{'=' * 70}")
    print(f"  Per-Class Threshold Optimization (validation set)")
    print(f"{'=' * 70}")
    print(f"  {'Class':<22s} {'Best Thr':>9s} {'Val F1':>8s} {'F1@0.5':>8s} {'Delta':>7s}")
    print(f"  {'-' * 55}")

    for i, name in enumerate(class_names):
        best_f1 = 0.0
        best_thr = 0.5
        f1_at_05 = f1_score(
            targets[:, i], (probs[:, i] >= 0.5).astype(int), zero_division=0
        )

        for thr in thresholds_to_try:
            preds = (probs[:, i] >= thr).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = round(float(thr), 2)

        optimal[name] = {
            "threshold": best_thr,
            "val_f1": round(best_f1, 4),
            "val_f1_at_0.5": round(f1_at_05, 4),
        }

        delta = best_f1 - f1_at_05
        marker = "+" if delta > 0.01 else ("~" if delta > -0.01 else "-")
        print(
            f"  {name:<22s} {best_thr:>9.2f} {best_f1:>8.4f} "
            f"{f1_at_05:>8.4f} {delta:>+7.4f} {marker}"
        )

    return optimal


def evaluate_with_thresholds(probs, targets, class_names, thresholds):
    """Evaluate macro metrics using given per-class thresholds."""
    thr_array = np.array([thresholds[name] for name in class_names])
    preds = (probs >= thr_array[np.newaxis, :]).astype(int)

    metrics = {}

    # Per-class F1, precision, recall
    per_class_f1 = []
    per_class_prec = []
    per_class_rec = []
    for i, name in enumerate(class_names):
        f1 = f1_score(targets[:, i], preds[:, i], zero_division=0)
        prec = precision_score(targets[:, i], preds[:, i], zero_division=0)
        rec = recall_score(targets[:, i], preds[:, i], zero_division=0)
        metrics[f"f1_{name}"] = round(f1, 4)
        metrics[f"precision_{name}"] = round(prec, 4)
        metrics[f"recall_{name}"] = round(rec, 4)
        per_class_f1.append(f1)
        per_class_prec.append(prec)
        per_class_rec.append(rec)

    metrics["macro_f1"] = round(np.mean(per_class_f1), 4)
    metrics["macro_precision"] = round(np.mean(per_class_prec), 4)
    metrics["macro_recall"] = round(np.mean(per_class_rec), 4)

    # AUC is threshold-independent (same for both)
    per_class_auc = []
    for i in range(len(class_names)):
        try:
            auc = roc_auc_score(targets[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")
        per_class_auc.append(auc)
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    metrics["macro_auc"] = round(np.mean(valid_aucs), 4) if valid_aucs else 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Per-class threshold optimization"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt"
    )
    args = parser.parse_args()

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[threshold] Device: {device}")

    class_names = cfg["dataset"]["class_names"]

    # --- Load model from checkpoint ---
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"[threshold] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
        f"macro-AUC {ckpt.get('macro_auc', '?')}"
    )

    # --- Build val and test datasets ---
    manifest_dir = cfg["paths"]["manifest_dir"]
    val_csv = os.path.join(manifest_dir, "manifest_val.csv")
    test_csv = os.path.join(manifest_dir, "manifest_test.csv")

    val_ds = ChestXrayDataset(val_csv, cfg, split="val")

    test_ds = ChestXrayDataset(test_csv, cfg, split="test")

    bs = cfg["training"]["batch_size"]
    nw = min(cfg["training"]["num_workers"], 2)  # conservative for Windows

    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    # ===================================================================
    # Step 1a: Collect VALIDATION predictions (threshold selection happens here)
    # ===================================================================
    print(f"\n[threshold] Running inference on VALIDATION set ({len(val_ds)} samples)...")
    val_probs, val_targets = collect_predictions(model, val_loader, device)

    # ===================================================================
    # Step 1b: Optimize thresholds on validation data
    # ===================================================================
    optimal = optimize_thresholds_per_class(val_probs, val_targets, class_names)

    # Save thresholds
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    thr_path = os.path.join(cfg["paths"]["output_dir"], "optimized_thresholds.json")
    with open(thr_path, "w") as f:
        json.dump(optimal, f, indent=2)
    print(f"\n[threshold] Optimized thresholds saved -> {thr_path}")

    # ===================================================================
    # Step 1c: Evaluate on TEST set with LOCKED thresholds from val
    # ===================================================================
    print(f"\n[threshold] Running inference on TEST set ({len(test_ds)} samples)...")
    test_probs, test_targets = collect_predictions(model, test_loader, device)

    default_thresholds = {name: 0.5 for name in class_names}
    optimized_thresholds = {name: optimal[name]["threshold"] for name in class_names}

    default_metrics = evaluate_with_thresholds(
        test_probs, test_targets, class_names, default_thresholds
    )
    optimized_metrics = evaluate_with_thresholds(
        test_probs, test_targets, class_names, optimized_thresholds
    )

    # --- Print comparison ---
    print(f"\n{'=' * 70}")
    print(f"  TEST SET -- Before / After Threshold Optimization")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<22s} {'Default 0.5':>12s} {'Optimized':>12s} {'Delta':>10s}")
    print(f"  {'-' * 58}")
    for metric_name in ["macro_auc", "macro_f1", "macro_precision", "macro_recall"]:
        d = default_metrics[metric_name]
        o = optimized_metrics[metric_name]
        delta = o - d
        marker = "+" if delta > 0.005 else ("-" if delta < -0.005 else "~")
        print(
            f"  {metric_name:<22s} {d:>12.4f} {o:>12.4f} {delta:>+10.4f} {marker}"
        )

    print(f"\n  Per-Class F1 (test set):")
    print(f"  {'Class':<22s} {'F1@0.5':>8s} {'F1@opt':>8s} {'Thr':>6s} {'Delta':>8s}")
    print(f"  {'-' * 55}")
    for name in class_names:
        d = default_metrics[f"f1_{name}"]
        o = optimized_metrics[f"f1_{name}"]
        thr = optimized_thresholds[name]
        delta = o - d
        marker = "+" if delta > 0.005 else ("-" if delta < -0.005 else "~")
        print(
            f"  {name:<22s} {d:>8.4f} {o:>8.4f} {thr:>6.2f} {delta:>+8.4f} {marker}"
        )

    # --- Save full results ---
    results = {
        "default_0.5": default_metrics,
        "optimized": optimized_metrics,
        "thresholds": optimized_thresholds,
    }
    results_path = os.path.join(
        cfg["paths"]["output_dir"], "threshold_comparison.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[threshold] Full comparison saved -> {results_path}")
    print("[threshold] Done.")


if __name__ == "__main__":
    main()
