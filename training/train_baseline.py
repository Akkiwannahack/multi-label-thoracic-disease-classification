"""
train_baseline.py — Ablation: Train the ResNet50-only baseline
"""

import argparse
import csv
import os

import torch
from torch.amp import GradScaler
import yaml

from models.baseline_resnet50 import build_baseline_model
from models.heads import count_parameters, parameter_summary, verify_no_softmax
from training.losses import build_loss
from training.train import train_one_epoch, validate


def train_baseline_full(cfg: dict, device: torch.device) -> None:
    print("\n" + "=" * 60)
    print("  Baseline Training — ResNet50 Only (no Vision Mamba)")
    print("=" * 60)

    model = build_baseline_model(cfg).to(device)
    print(f"[baseline] Total params:     {count_parameters(model, only_trainable=False):,}")
    print(f"[baseline] Trainable params: {count_parameters(model, only_trainable=True):,}")
    for name, info in parameter_summary(model).items():
        print(f"  {name}: {info['total']:,} total, {info['trainable']:,} trainable")

    assert verify_no_softmax(model), "Softmax detected in multi-label model!"

    from data.dataset import build_dataloaders
    loaders = build_dataloaders(cfg)
    assert "train" in loaders and "val" in loaders

    balance_path = os.path.join(cfg["paths"]["manifest_dir"], "class_balance.json")
    criterion = build_loss(
        cfg,
        class_balance_path=balance_path if os.path.isfile(balance_path) else None,
    ).to(device)

    t = cfg["training"]
    class_names = cfg["dataset"]["class_names"]
    num_epochs = t["epochs"]
    patience = t["early_stopping_patience"]
    use_amp = t["mixed_precision"] and device.type == "cuda"

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

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    metrics_path = os.path.join(cfg["paths"]["output_dir"], "metrics_log_baseline.csv")
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "baseline_best_model.pt")
    best_macro_auc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        if epoch == t["stage1_epochs"] + 1:
            print(f"\n[baseline] Stage 2: unfreezing all ResNet layers, reducing backbone LR")
            model.unfreeze_all()
            param_groups = model.get_param_groups(
                lr_backbone=t["lr_backbone_stage2"],
                lr_head=t["lr_head_stage2"],
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

        print(f"\nEpoch {epoch}/{num_epochs}")

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

        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = round(train_loss, 4)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"macro_auc={val_metrics['macro_auc']:.4f}  "
              f"macro_f1={val_metrics['macro_f1']:.4f}")

        write_header = (epoch == 1) or not os.path.isfile(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(val_metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(val_metrics)

        if val_metrics["macro_auc"] > best_macro_auc:
            best_macro_auc = val_metrics["macro_auc"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_auc": best_macro_auc,
                "config": cfg,
            }, ckpt_path)
            print(f"  ★ New best macro-AUC: {best_macro_auc:.4f} — saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\n[baseline] Early stopping — no improvement for {patience} epochs.")
            break

    print(f"\n[baseline] Training complete. Best macro-AUC: {best_macro_auc:.4f}")
    print(f"[baseline] Metrics log: {metrics_path}")
    print(f"[baseline] Checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50-only baseline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] Device: {device}")

    train_baseline_full(cfg, device)


if __name__ == "__main__":
    main()
