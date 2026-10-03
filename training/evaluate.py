"""
evaluate.py — Phase 10: Evaluation & Baseline Comparison

Evaluates the trained model on the held-out test set (or validation set)
and optionally compares against a ResNet50-only baseline.

Fix E: Baseline evaluation uses build_baseline_model (not build_model).
       Validates checkpoint model_type.
Fix H: --split flag (val or test), --thresholds flag for fitted thresholds,
       deterministic eval-mode inference, per-run directories.

Usage:
    python -m training.evaluate --config configs/base.yaml \\
        --checkpoint checkpoints/best_model.pt

    python -m training.evaluate --config configs/base.yaml \\
        --checkpoint checkpoints/best_model.pt --split val

    python -m training.evaluate --config configs/base.yaml \\
        --checkpoint checkpoints/best_model.pt \\
        --thresholds outputs/optimized_thresholds.json
"""

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.hybrid_model import build_model
from models.baseline_resnet50 import build_baseline_model
from training.train import compute_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    thresholds: dict[str, float] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run full evaluation on a DataLoader.

    Fix H: deterministic eval-mode inference, optional per-class thresholds.

    Returns:
        (metrics_dict, all_probs, all_targets)
    """
    model.eval()
    all_probs = []
    all_targets = []

    for images, labels in tqdm(loader, desc="  evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(all_probs, all_targets, class_names)

    # Use provided thresholds or default 0.5
    if thresholds is not None:
        thr_array = np.array([thresholds[name] for name in class_names])
    else:
        thr_array = np.full(len(class_names), 0.5)

    preds = (all_probs >= thr_array[np.newaxis, :]).astype(int)

    # Per-class precision/recall at the chosen thresholds
    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = round(
            precision_score(all_targets[:, i], preds[:, i], zero_division=0), 4
        )
        metrics[f"recall_{name}"] = round(
            recall_score(all_targets[:, i], preds[:, i], zero_division=0), 4
        )
        metrics[f"f1_{name}"] = round(
            f1_score(all_targets[:, i], preds[:, i], zero_division=0), 4
        )

    # Recompute macro F1/precision/recall at the chosen thresholds
    if thresholds is not None:
        metrics["macro_f1_thresholded"] = round(
            np.mean([metrics[f"f1_{name}"] for name in class_names]), 4
        )
        metrics["macro_precision_thresholded"] = round(
            np.mean([metrics[f"precision_{name}"] for name in class_names]), 4
        )
        metrics["macro_recall_thresholded"] = round(
            np.mean([metrics[f"recall_{name}"] for name in class_names]), 4
        )

    return metrics, all_probs, all_targets


def print_evaluation_report(
    metrics: dict,
    class_names: list[str],
    model_name: str = "Hybrid ResNet50+Mamba",
    thresholds: dict[str, float] | None = None,
) -> None:
    """Pretty-print the evaluation results."""
    print(f"\n{'='*70}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*70}")

    print(f"\n  Overall Accuracy:  {metrics.get('overall_accuracy', 'N/A'):.4f}")
    print(f"  Macro AUC-ROC:     {metrics['macro_auc']:.4f}")
    print(f"  Micro AUC-ROC:     {metrics.get('micro_auc', 'N/A'):.4f}")
    print(f"  Macro F1 @0.5:     {metrics['macro_f1']:.4f}")
    if "macro_f1_thresholded" in metrics:
        print(f"  Macro F1 @fitted:  {metrics['macro_f1_thresholded']:.4f}")
    print(f"  Micro F1:          {metrics['micro_f1']:.4f}")
    print(f"  Macro Precision:   {metrics.get('macro_precision', 'N/A'):.4f}")
    print(f"  Macro Recall:      {metrics.get('macro_recall', 'N/A'):.4f}")

    thr_label = "Thr" if thresholds else ""
    print(f"\n  {'Class':<22s} {'AUC':>7s} {'Precision':>10s} {'Recall':>8s} {'F1':>7s} {thr_label:>5s}")
    print(f"  {'-'*60}")
    for name in class_names:
        auc = metrics.get(f"auc_{name}", float("nan"))
        prec = metrics.get(f"precision_{name}", float("nan"))
        rec = metrics.get(f"recall_{name}", float("nan"))
        f1 = metrics.get(f"f1_{name}", float("nan"))
        thr = f"{thresholds[name]:.2f}" if thresholds else ""
        print(f"  {name:<22s} {auc:>7.4f} {prec:>10.4f} {rec:>8.4f} {f1:>7.4f} {thr:>5s}")


def save_comparison_table(
    results: list[dict],
    output_path: str,
) -> None:
    """Save a baseline comparison CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n[eval] Comparison table saved -> {output_path}")


def plot_roc_curves(all_targets: np.ndarray, all_probs: np.ndarray, class_names: list[str], output_path: str):
    """Plot ROC curve for each class and save to a PNG file."""
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_probs[:, i])
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.3f})")
        except ValueError:
            pass # Skip classes with only one label value

    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[eval] ROC curves saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate thoracic disease classifier")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--baseline-checkpoint", type=str, default=None,
                        help="Path to ResNet50-only baseline checkpoint for ablation comparison")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Which split to evaluate on (Fix H)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Path to optimized_thresholds.json (Fix H)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["dataset"]["class_names"]

    # Build loaders
    from data.dataset import build_dataloaders
    loaders = build_dataloaders(cfg)
    assert args.split in loaders, f"No {args.split} manifest found — run build_manifest.py first."

    eval_loader = loaders[args.split]

    # Load thresholds if provided
    thresholds = None
    if args.thresholds:
        if not os.path.isfile(args.thresholds):
            raise FileNotFoundError(f"Thresholds file not found: {args.thresholds}")
        with open(args.thresholds, "r") as f:
            thr_data = json.load(f)
        # Handle both formats: {"class": {"threshold": x}} and {"class": x}
        thresholds = {}
        for name in class_names:
            if name not in thr_data:
                raise KeyError(f"Threshold for '{name}' not found in {args.thresholds}")
            val = thr_data[name]
            thresholds[name] = val["threshold"] if isinstance(val, dict) else val

    comparison_results = []

    # ---- Evaluate hybrid model ----
    print(f"\n[eval] Loading hybrid model checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Fix E: validate checkpoint model type
    model_type = ckpt.get("model_type", "hybrid_resnet_mamba")
    prov = ckpt.get("model_provenance", {})
    if prov:
        model_type = prov.get("model_type", model_type)

    if model_type == "baseline_resnet50":
        raise ValueError(
            f"Checkpoint at {args.checkpoint} is a baseline model. "
            f"Use --baseline-checkpoint instead."
        )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[eval] Loaded from epoch {ckpt.get('epoch', '?')}, "
          f"val macro-AUC {ckpt.get('macro_auc', ckpt.get('best_macro_auc', '?'))}")

    metrics, all_probs, all_targets = evaluate_model(
        model, eval_loader, device, class_names, thresholds
    )
    print_evaluation_report(metrics, class_names,
                           f"Hybrid ResNet50 + Vision Mamba ({args.split})",
                           thresholds)

    # Plot ROC curves
    roc_path = os.path.join(cfg["paths"]["output_dir"], f"roc_curves_{args.split}.png")
    plot_roc_curves(all_targets, all_probs, class_names, roc_path)

    comparison_results.append({
        "model": "ResNet50 + Vision Mamba (Ours)",
        "split": args.split,
        "overall_accuracy": round(metrics.get("overall_accuracy", 0), 4),
        "macro_auc": round(metrics["macro_auc"], 4),
        "micro_auc": round(metrics.get("micro_auc", 0), 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "micro_f1": round(metrics["micro_f1"], 4),
        "macro_precision": round(metrics.get("macro_precision", 0), 4),
        "macro_recall": round(metrics.get("macro_recall", 0), 4),
    })

    # ---- Optionally evaluate ResNet50-only baseline (Fix E) ----
    if args.baseline_checkpoint and os.path.isfile(args.baseline_checkpoint):
        print("\n[eval] Loading ResNet50-only baseline checkpoint...")
        # Fix E: use build_baseline_model, not build_model
        baseline_model = build_baseline_model(cfg).to(device)
        baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])

        baseline_metrics, _, _ = evaluate_model(
            baseline_model, eval_loader, device, class_names, thresholds
        )
        print_evaluation_report(baseline_metrics, class_names,
                               f"ResNet50 Only (Baseline) ({args.split})",
                               thresholds)

        comparison_results.append({
            "model": "ResNet50 Only (Baseline)",
            "split": args.split,
            "overall_accuracy": round(baseline_metrics.get("overall_accuracy", 0), 4),
            "macro_auc": round(baseline_metrics["macro_auc"], 4),
            "micro_auc": round(baseline_metrics.get("micro_auc", 0), 4),
            "macro_f1": round(baseline_metrics["macro_f1"], 4),
            "micro_f1": round(baseline_metrics["micro_f1"], 4),
            "macro_precision": round(baseline_metrics.get("macro_precision", 0), 4),
            "macro_recall": round(baseline_metrics.get("macro_recall", 0), 4),
        })

    # Save comparison
    output_path = os.path.join(cfg["paths"]["output_dir"], f"baseline_comparison_{args.split}.csv")
    save_comparison_table(comparison_results, output_path)


if __name__ == "__main__":
    main()
