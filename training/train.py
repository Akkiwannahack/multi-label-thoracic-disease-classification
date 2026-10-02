"""
train.py — Phase 4 & 5: Training Pipeline

Includes:
  - Overfit sanity check (Phase 4) — verifies the model can memorize 32 images
  - Full training loop (Phase 5) — staged fine-tuning with mixed precision,
    early stopping, and comprehensive metric logging

Usage:
    # Overfit sanity check
    python -m training.train --config configs/base.yaml --overfit

    # Full training
    python -m training.train --config configs/base.yaml
"""

import argparse
import csv
import tempfile
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from models.hybrid_model import build_model, HybridResNetMamba
from models.heads import count_parameters, parameter_summary, verify_no_softmax
from training.losses import build_loss


# ===================== Metrics =============================================

def compute_metrics(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute per-class AUC-ROC, macro/micro AUC, macro/micro F1.

    Args:
        all_probs:   (N, 14) sigmoid probabilities
        all_targets: (N, 14) binary ground truth

    Returns:
        Dictionary of metrics
    """
    num_classes = all_probs.shape[1]
    metrics = {}

    # Per-class AUC-ROC
    per_class_auc = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            auc = float("nan")  # class has only one label value in this split
        per_class_auc.append(auc)
        metrics[f"auc_{class_names[i]}"] = round(auc, 4)

    # Macro / micro AUC
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    metrics["macro_auc"] = round(np.mean(valid_aucs), 4) if valid_aucs else 0.0

    try:
        metrics["micro_auc"] = round(
            roc_auc_score(all_targets, all_probs, average="micro"), 4
        )
    except ValueError:
        metrics["micro_auc"] = 0.0

    # Binary predictions at threshold 0.5
    preds = (all_probs >= 0.5).astype(int)

    # Per-label accuracy (fraction of individual label predictions that are correct)
    metrics["overall_accuracy"] = round(
        float((preds == all_targets).mean()), 4
    )

    # Macro F1 / Micro F1
    metrics["macro_f1"] = round(
        f1_score(all_targets, preds, average="macro", zero_division=0), 4
    )
    metrics["micro_f1"] = round(
        f1_score(all_targets, preds, average="micro", zero_division=0), 4
    )

    # Macro Precision / Macro Recall
    metrics["macro_precision"] = round(
        precision_score(all_targets, preds, average="macro", zero_division=0), 4
    )
    metrics["macro_recall"] = round(
        recall_score(all_targets, preds, average="macro", zero_division=0), 4
    )

    return metrics


# ===================== Training Loop =======================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
    accumulation_steps: int = 4,
) -> float:
    """Train for one epoch with gradient accumulation + clipping; return average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  train", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(images)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

    # Flush remaining gradients if last batch didn't complete a full cycle
    if n_batches > 0 and (batch_idx + 1) % accumulation_steps != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    use_amp: bool = False,
) -> tuple[float, dict]:
    """Validate; return (avg_loss, metrics_dict)."""
    model.eval()
    running_loss = 0.0
    n_batches = 0

    all_probs = []
    all_targets = []

    for images, labels in tqdm(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / max(n_batches, 1)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(all_probs, all_targets, class_names)
    metrics["val_loss"] = round(avg_loss, 4)

    return avg_loss, metrics


# ===================== Overfit Sanity Check (Phase 4) ======================

def overfit_sanity_check(cfg: dict, device: torch.device) -> bool:
    """Phase 4: Overfit the model on 32 images for 50 epochs.

    This catches label-encoding bugs, wrong loss reduction, or frozen-gradient
    bugs before burning GPU hours on the full run.

    Returns True if the test passes (training loss → ~0).
    """
    print("\n" + "=" * 60)
    print("  Phase 4 — Overfit Sanity Check (32 images, 50 epochs)")
    print("=" * 60)

    model = build_model(cfg).to(device)
    num_classes = cfg["model"]["num_classes"]

    # Verify no softmax
    assert verify_no_softmax(model), "Softmax detected in multi-label model!"

    # Build a tiny synthetic dataset for the overfit test
    # (If real manifests are available, use them; otherwise use random data)
    manifest_dir = cfg["paths"]["manifest_dir"]
    train_csv = os.path.join(manifest_dir, "manifest_train.csv")

    if os.path.isfile(train_csv):
        from data.dataset import ChestXrayDataset
        full_ds = ChestXrayDataset(train_csv, cfg, split="train")
        # Take first 32 samples
        tiny_ds = torch.utils.data.Subset(full_ds, list(range(min(32, len(full_ds)))))
    else:
        print("[overfit] No manifest found — using synthetic random data.")
        images = torch.randn(32, 3, 224, 224)
        labels = torch.randint(0, 2, (32, num_classes)).float()
        tiny_ds = torch.utils.data.TensorDataset(images, labels)

    tiny_loader = torch.utils.data.DataLoader(tiny_ds, batch_size=8, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(50):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tiny_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Exact-match multi-label accuracy
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/50  loss={epoch_loss/len(tiny_loader):.4f}  "
                  f"exact-match-acc={acc:.3f}")

    final_acc = correct / total if total > 0 else 0
    passed = final_acc >= 0.90
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n  Overfit test: exact-match accuracy = {final_acc:.3f} — {status}")

    if not passed:
        print("  WARNING: Model cannot overfit 32 images — check labels, loss, and gradients!")

    return passed


# ===================== Stage Builders ======================================

def _build_stage1(model, cfg):
    """Freeze early layers & return stage-1 optimizer + scheduler + scaler."""
    t = cfg["training"]
    use_amp = t["mixed_precision"] and torch.cuda.is_available()

    model.freeze_early_layers()
    param_groups = model.get_param_groups(
        lr_backbone=t["lr_backbone_stage1"],
        lr_head=t["lr_head_stage1"],
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=t.get("weight_decay", 5e-4))

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
    return optimizer, scheduler, scaler


def _build_stage2(model, cfg):
    """Unfreeze all & return stage-2 optimizer + warmup→cosine scheduler + scaler."""
    t = cfg["training"]
    use_amp = t["mixed_precision"] and torch.cuda.is_available()
    warmup_epochs = t.get("warmup_epochs_stage2", 2)

    model.unfreeze_all()
    param_groups = model.get_param_groups(
        lr_backbone=t["lr_backbone_stage2"],
        lr_head=t["lr_head_stage2"],
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=t.get("weight_decay", 5e-4))

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )

    if t["scheduler"] == "cosine_warm_restarts":
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t["cosine_T_0"], T_mult=t["cosine_T_mult"]
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        cosine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=t["plateau_factor"],
            patience=t["plateau_patience"]
        )
        scheduler = warmup_scheduler
        scheduler._plateau_fallback = cosine_scheduler
        scheduler._warmup_remaining = warmup_epochs

    scaler = GradScaler(device="cuda") if use_amp else None
    return optimizer, scheduler, scaler


def _save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                     best_macro_auc, epochs_without_improvement):
    """Atomically save a checkpoint via write-to-temp + os.replace()."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_macro_auc": best_macro_auc,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ===================== Full Training (Phase 5) =============================

def train_full(cfg: dict, device: torch.device, resume: bool = False) -> None:
    """Phase 5: Full training with staged fine-tuning, mixed precision, and early stopping."""
    print("\n" + "=" * 60)
    print("  Phase 5 — Full Training")
    print("=" * 60)

    # --- Build model ---
    model = build_model(cfg).to(device)
    print(f"[train] Total params:     {count_parameters(model, only_trainable=False):,}")
    print(f"[train] Trainable params: {count_parameters(model, only_trainable=True):,}")
    for name, info in parameter_summary(model).items():
        print(f"  {name}: {info['total']:,} total, {info['trainable']:,} trainable")

    # --- Build dataloaders ---
    from data.dataset import build_dataloaders
    loaders = build_dataloaders(cfg)
    assert "train" in loaders and "val" in loaders, \
        "Missing train/val dataloaders — run build_manifest.py first."

    # --- Loss ---
    balance_path = os.path.join(cfg["paths"]["manifest_dir"], "class_balance.json")
    criterion = build_loss(
        cfg,
        class_balance_path=balance_path if os.path.isfile(balance_path) else None,
    ).to(device)

    # --- Training config ---
    t = cfg["training"]
    class_names = cfg["dataset"]["class_names"]
    num_epochs = t["epochs"]
    patience = t["early_stopping_patience"]
    use_amp = t["mixed_precision"] and device.type == "cuda"

    # --- Stage 1: freeze early ResNet layers ---
    optimizer, scheduler, scaler = _build_stage1(model, cfg)

    start_epoch = 1
    best_macro_auc = 0.0
    epochs_without_improvement = 0

    # --- Metric logging ---
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    metrics_path = os.path.join(cfg["paths"]["output_dir"], "metrics_log.csv")

    # --- Resume from checkpoint ---
    if resume:
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "last_checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        resumed_epoch = ckpt["epoch"]
        start_epoch = resumed_epoch + 1
        best_macro_auc = ckpt["best_macro_auc"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]

        # Rebuild the correct stage BEFORE loading optimizer/scheduler state
        if resumed_epoch >= t["stage1_epochs"]:
            # We are in (or past) stage 2 → rebuild stage-2 objects
            optimizer, scheduler, scaler = _build_stage2(model, cfg)
        # else: stage-1 objects already built above

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        print(f"[train] Resumed from epoch {resumed_epoch} "
              f"(best_macro_auc={best_macro_auc:.4f})")
    else:
        # Fresh run — wipe metrics_log.csv so header is written cleanly
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs + 1):
        # Stage transition: unfreeze all after stage1_epochs
        if epoch == t["stage1_epochs"] + 1:
            print(f"\n[train] Stage 2: unfreezing all ResNet layers")
            optimizer, scheduler, scaler = _build_stage2(model, cfg)

        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        accumulation_steps = t.get("accumulation_steps", 4)
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp,
            accumulation_steps=accumulation_steps,
        )

        # Validate
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device, class_names, use_amp
        )

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["macro_auc"])
        else:
            scheduler.step()

        # Log
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = round(train_loss, 4)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_metrics['overall_accuracy']:.4f}  "
              f"macro_auc={val_metrics['macro_auc']:.4f}  "
              f"macro_f1={val_metrics['macro_f1']:.4f}")

        # Save metrics CSV
        write_header = not os.path.isfile(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(val_metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(val_metrics)

        # Best model checkpoint
        if val_metrics["macro_auc"] > best_macro_auc:
            best_macro_auc = val_metrics["macro_auc"]
            epochs_without_improvement = 0
            ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
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

        # Save resumable checkpoint (every epoch, atomic)
        last_ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "last_checkpoint.pt")
        _save_checkpoint(
            last_ckpt_path, epoch, model, optimizer, scheduler, scaler,
            best_macro_auc, epochs_without_improvement,
        )
        print(f"  \U0001f4be Saved last_checkpoint.pt (epoch {epoch})")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n[train] Early stopping — no improvement for {patience} epochs.")
            break

    print(f"\n[train] Training complete. Best macro-AUC: {best_macro_auc:.4f}")
    print(f"[train] Metrics log: {metrics_path}")


# ===================== Entry Point =========================================

def main():
    parser = argparse.ArgumentParser(description="Train thoracic disease classifier")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--overfit", action="store_true",
                        help="Run Phase 4 overfit sanity check only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoints/last_checkpoint.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    if args.overfit:
        passed = overfit_sanity_check(cfg, device)
        if not passed:
            print("\n⚠ Overfit test FAILED — do not proceed to full training!")
            exit(1)
    else:
        train_full(cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
