"""
evaluate.py — Phase 10: Evaluation & Baseline Comparison

Evaluates the trained model on the held-out test set and optionally
compares against a ResNet50-only baseline (ablation).

Usage:
    python -m training.evaluate --config configs/base.yaml \
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import csv
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
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.hybrid_model import build_model
from training.train import compute_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run full evaluation on a DataLoader.

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

    # Additional per-class precision/recall
    preds = (all_probs >= 0.5).astype(int)
    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = round(
            precision_score(all_targets[:, i], preds[:, i], zero_division=0), 4
        )
        metrics[f"recall_{name}"] = round(
            recall_score(all_targets[:, i], preds[:, i], zero_division=0), 4
        )

    return metrics, all_probs, all_targets


def print_evaluation_report(
    metrics: dict,
    class_names: list[str],
    model_name: str = "Hybrid ResNet50+Mamba",
) -> None:
    """Pretty-print the evaluation results."""
    print(f"\n{'='*70}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*70}")

    print(f"\n  Overall Accuracy:  {metrics.get('overall_accuracy', 'N/A')}")
    print(f"  Macro AUC-ROC:     {metrics['macro_auc']:.4f}")
    print(f"  Micro AUC-ROC:     {metrics.get('micro_auc', 'N/A')}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:          {metrics['micro_f1']:.4f}")
    print(f"  Macro Precision:   {metrics.get('macro_precision', 'N/A')}")
    print(f"  Macro Recall:      {metrics.get('macro_recall', 'N/A')}")

    print(f"\n  {'Class':<22s} {'AUC':>7s} {'Precision':>10s} {'Recall':>8s}")
    print(f"  {'-'*50}")
    for name in class_names:
        auc = metrics.get(f"auc_{name}", float("nan"))
        prec = metrics.get(f"precision_{name}", float("nan"))
        rec = metrics.get(f"recall_{name}", float("nan"))
        print(f"  {name:<22s} {auc:>7.4f} {prec:>10.4f} {rec:>8.4f}")


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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["dataset"]["class_names"]

    # Build test loader
    from data.dataset import build_dataloaders
    loaders = build_dataloaders(cfg)
    assert "test" in loaders, "No test manifest found — run build_manifest.py first."

    comparison_results = []

    # ---- Evaluate hybrid model ----
    print("\n[eval] Loading hybrid model checkpoint...")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[eval] Loaded from epoch {ckpt.get('epoch', '?')}, "
          f"val macro-AUC {ckpt.get('macro_auc', '?')}")

    metrics, all_probs, all_targets = evaluate_model(model, loaders["test"], device, class_names)
    print_evaluation_report(metrics, class_names, "Hybrid ResNet50 + Vision Mamba")
    
    # Plot ROC curves
    roc_path = os.path.join(cfg["paths"]["output_dir"], "roc_curves.png")
    plot_roc_curves(all_targets, all_probs, class_names, roc_path)

    comparison_results.append({
        "model": "ResNet50 + Vision Mamba (Ours)",
        "overall_accuracy": metrics.get("overall_accuracy", ""),
        "macro_auc": metrics["macro_auc"],
        "micro_auc": metrics.get("micro_auc", ""),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "macro_precision": metrics.get("macro_precision", ""),
        "macro_recall": metrics.get("macro_recall", ""),
    })

    # ---- Optionally evaluate ResNet50-only baseline ----
    if args.baseline_checkpoint and os.path.isfile(args.baseline_checkpoint):
        print("\n[eval] Loading ResNet50-only baseline checkpoint...")
        baseline_model = build_model(cfg).to(device)
        baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])

        baseline_metrics, _, _ = evaluate_model(
            baseline_model, loaders["test"], device, class_names
        )
        print_evaluation_report(baseline_metrics, class_names, "ResNet50 Only (Baseline)")

        comparison_results.append({
            "model": "ResNet50 Only (Baseline)",
            "overall_accuracy": baseline_metrics.get("overall_accuracy", ""),
            "macro_auc": baseline_metrics["macro_auc"],
            "micro_auc": baseline_metrics.get("micro_auc", ""),
            "macro_f1": baseline_metrics["macro_f1"],
            "micro_f1": baseline_metrics["micro_f1"],
            "macro_precision": baseline_metrics.get("macro_precision", ""),
            "macro_recall": baseline_metrics.get("macro_recall", ""),
        })

    # Save comparison
    output_path = os.path.join(cfg["paths"]["output_dir"], "baseline_comparison.csv")
    save_comparison_table(comparison_results, output_path)


if __name__ == "__main__":
    main()
