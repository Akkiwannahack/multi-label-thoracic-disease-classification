"""
train.py — Phase 4 & 5: Training Pipeline

Includes:
  - Overfit sanity check (Phase 4) — verifies the model can memorize 32 images
  - Full training loop (Phase 5) — staged fine-tuning with mixed precision,
    early stopping, and comprehensive metric logging

Fix B: Frozen backbone + BN stats, explicit param groups, optimizer state
       preserved across stage change.
Fix E: train_one_epoch returns one float (matching callers), shared return
       contract.
Fix F: Phase-aware resume, warmup→plateau schedule, phase-aware early
       stopping, accumulation tail normalization, seeding, full-precision
       metric selection.

Usage:
    # Overfit sanity check
    python -m training.train --config configs/base.yaml --overfit

    # Full training
    python -m training.train --config configs/base.yaml
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import tempfile
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
from models.resnet_backbone import ResNet50Backbone
from models.heads import count_parameters, parameter_summary, verify_no_softmax
from training.losses import build_loss

logger = logging.getLogger(__name__)


# ===================== Seeding (Fix F) =====================================

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> dict:
    """Capture current RNG states for checkpoint."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    """Restore RNG states from checkpoint."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.random.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


# ===================== Metrics =============================================

def compute_metrics(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute per-class AUC-ROC, macro/micro AUC, macro/micro F1.

    Fix F: returns full-precision values for selection; rounds only for display.

    Args:
        all_probs:   (N, 14) sigmoid probabilities
        all_targets: (N, 14) binary ground truth

    Returns:
        Dictionary of metrics (full precision floats)
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
        metrics[f"auc_{class_names[i]}"] = auc

    # Macro / micro AUC
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    metrics["macro_auc"] = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    try:
        metrics["micro_auc"] = float(
            roc_auc_score(all_targets, all_probs, average="micro")
        )
    except ValueError:
        metrics["micro_auc"] = 0.0

    # Binary predictions at threshold 0.5
    preds = (all_probs >= 0.5).astype(int)

    # Per-label accuracy (fraction of individual label predictions that are correct)
    metrics["overall_accuracy"] = float((preds == all_targets).mean())

    # Macro F1 / Micro F1
    metrics["macro_f1"] = float(
        f1_score(all_targets, preds, average="macro", zero_division=0)
    )
    metrics["micro_f1"] = float(
        f1_score(all_targets, preds, average="micro", zero_division=0)
    )

    # Macro Precision / Macro Recall
    metrics["macro_precision"] = float(
        precision_score(all_targets, preds, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(all_targets, preds, average="macro", zero_division=0)
    )

    return metrics


def round_metrics(metrics: dict) -> dict:
    """Round metrics to 4 decimal places for display only."""
    return {k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()}


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
    freeze_bn: bool = False,
) -> float:
    """Train for one epoch with gradient accumulation + clipping; return average loss.

    Fix B: freeze_bn parameter to keep BN running stats frozen.
    Fix E: Returns ONE float (average loss). All callers expect this.
    Fix F: Accumulation tail normalized by actual micro-batch count in the window.
    """
    model.train()
    if freeze_bn:
        ResNet50Backbone.freeze_bn_stats(model)

    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    # Track micro-batches in current accumulation window
    micro_in_window = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        micro_in_window += 1

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
                micro_in_window = 0
        else:
            logits = model(images)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_in_window = 0

        running_loss += loss.item() * accumulation_steps
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

    # Fix F: Flush remaining gradients — normalize by actual micro-batches
    if n_batches > 0 and micro_in_window > 0:
        # Rescale: we divided each micro-loss by accumulation_steps,
        # but only micro_in_window of them accumulated. Scale correction:
        if micro_in_window != accumulation_steps:
            scale_correction = accumulation_steps / micro_in_window
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale_correction)

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
    metrics["val_loss"] = avg_loss

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


# ===================== Stage Builders (Fix B, F) ===========================

def _build_stage1(model, cfg):
    """Freeze backbone + BN, return optimizer + scheduler + scaler.

    Fix B: freeze entire backbone, keep BN stats frozen.
    """
    t = cfg["training"]
    use_amp = t["mixed_precision"] and torch.cuda.is_available()

    # Fix B: freeze entire backbone (not just early layers)
    model.freeze_backbone()

    # Build param groups for trainable params (mamba + classifier)
    fine_tune_layers = None  # no backbone layers in stage 1
    param_groups = model.get_param_groups_v2(
        lr_backbone=0.0,  # backbone is frozen, no LR needed
        lr_head=t.get("lr_head", t.get("lr_head_stage1", 1e-4)),
        fine_tune_layers=fine_tune_layers,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=t.get("weight_decay", 1e-4),
    )

    scheduler = _build_scheduler(optimizer, t, t.get("lr_head", t.get("lr_head_stage1", 1e-4)))
    scaler = GradScaler(device="cuda") if use_amp else None

    _log_param_groups(param_groups, "Stage 1")

    return optimizer, scheduler, scaler


def _build_stage2(model, cfg, prev_optimizer=None):
    """Unfreeze layer4, return optimizer + warmup→plateau scheduler + scaler.

    Fix B: unfreeze only layer4, keep BN stats frozen.
    Fix F: preserve optimizer state for already-trained params.
    """
    t = cfg["training"]
    use_amp = t["mixed_precision"] and torch.cuda.is_available()

    # Fix B: unfreeze layer4 only
    fine_tune_layers = t.get("fine_tune_layers", ["layer4"])
    model.unfreeze_layer4()

    lr_backbone = t.get("lr_backbone", t.get("lr_backbone_stage2", 1e-5))
    lr_head = t.get("lr_head", t.get("lr_head_stage2", 5e-5))

    param_groups = model.get_param_groups_v2(
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        fine_tune_layers=fine_tune_layers,
    )

    # Fix F: create new optimizer but transfer momentum for params
    # that were already being optimized
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=t.get("weight_decay", 1e-4),
    )

    if prev_optimizer is not None:
        _transfer_optimizer_state(prev_optimizer, optimizer)

    # Fix F: warmup → plateau schedule
    warmup_epochs = t.get("warmup_epochs_stage2", 2)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )

    scheduler_type = t.get("scheduler", "plateau")
    if scheduler_type == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=t.get("plateau_factor", 0.5),
            patience=t.get("plateau_patience", 1),
        )
        # Wrap: use warmup for first N epochs, then switch to plateau
        scheduler = WarmupThenPlateau(
            warmup_scheduler, plateau_scheduler, warmup_epochs
        )
    elif scheduler_type == "cosine_warm_restarts":
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t["cosine_T_0"], T_mult=t["cosine_T_mult"]
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = warmup_scheduler

    scaler = GradScaler(device="cuda") if use_amp else None

    _log_param_groups(param_groups, "Stage 2")

    return optimizer, scheduler, scaler


def _build_scheduler(optimizer, t, default_lr):
    """Build a scheduler for stage 1."""
    scheduler_type = t.get("scheduler", "plateau")
    if scheduler_type == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t["cosine_T_0"], T_mult=t["cosine_T_mult"]
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=t.get("plateau_factor", 0.5),
            patience=t.get("plateau_patience", 1),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class WarmupThenPlateau:
    """Fix F: Warmup for N epochs, then switch to ReduceLROnPlateau.

    Implements the step(val_metric) interface.
    """

    def __init__(self, warmup, plateau, warmup_epochs):
        self.warmup = warmup
        self.plateau = plateau
        self.warmup_epochs = warmup_epochs
        self._epoch = 0

    def step(self, val_metric=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            self.warmup.step()
        else:
            if val_metric is not None:
                self.plateau.step(val_metric)
            else:
                self.plateau.step(0.0)

    def state_dict(self):
        return {
            "warmup": self.warmup.state_dict(),
            "plateau": self.plateau.state_dict(),
            "warmup_epochs": self.warmup_epochs,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state):
        self.warmup.load_state_dict(state["warmup"])
        self.plateau.load_state_dict(state["plateau"])
        self.warmup_epochs = state["warmup_epochs"]
        self._epoch = state["epoch"]


def _transfer_optimizer_state(old_opt, new_opt):
    """Fix F: Transfer Adam momentum from old optimizer to new one
    for parameters that exist in both.
    """
    old_params = {}
    for group in old_opt.param_groups:
        for p in group["params"]:
            if id(p) in old_opt.state:
                old_params[id(p)] = old_opt.state[id(p)]

    for group in new_opt.param_groups:
        for p in group["params"]:
            pid = id(p)
            if pid in old_params:
                new_opt.state[p] = old_params[pid]

    transferred = sum(1 for g in new_opt.param_groups
                      for p in g["params"] if p in new_opt.state)
    logger.info("Transferred optimizer state for %d parameters", transferred)


def _log_param_groups(groups, stage_name):
    """Log parameter group info."""
    print(f"  [{stage_name}] Parameter groups:")
    for i, g in enumerate(groups):
        params = g["params"]
        n_params = sum(p.numel() for p in params)
        name = g.get("name", f"group_{i}")
        lr = g.get("lr", "default")
        # Sample first parameter name for debugging
        sample = ""
        for p in params:
            if p.numel() > 0:
                # Find the parameter name in the model
                sample = f" (e.g. {n_params} params)"
                break
        print(f"    {name}: lr={lr}, params={n_params:,}{sample}")


def _verify_param_groups(model, groups):
    """Fix B: Assert each trainable parameter is in exactly one group."""
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assigned = {}
    for i, g in enumerate(groups):
        name = g.get("name", f"group_{i}")
        for p in g["params"]:
            pid = id(p)
            if pid in assigned:
                raise RuntimeError(
                    f"Parameter in both '{assigned[pid]}' and '{name}'"
                )
            assigned[pid] = name

    missing = trainable - set(assigned.keys())
    if missing:
        logger.warning(
            "%d trainable parameters not in any optimizer group", len(missing)
        )


def _save_checkpoint(path, epoch, phase, model, optimizer, scheduler, scaler,
                     best_macro_auc, epochs_without_improvement, cfg):
    """Atomically save a checkpoint with full provenance (Fix D, F)."""
    state = {
        "epoch": epoch,
        "phase": phase,  # Fix F: save the actual phase
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_auc": best_macro_auc,
        "epochs_without_improvement": epochs_without_improvement,
        "config": cfg,
    }

    # Fix D: model provenance
    if hasattr(model, "get_provenance"):
        state["model_provenance"] = model.get_provenance()

    # Fix F: scheduler state
    if hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    # Fix F: RNG states
    state["rng_states"] = get_rng_states()

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
    """Phase 5: Full training with staged fine-tuning, mixed precision, and early stopping.

    Fix B: Frozen backbone + BN stats in stage 1, layer4 only in stage 2.
    Fix F: Phase-aware resume, warmup→plateau, phase-aware early stopping,
           seeding, full-precision selection.
    """
    print("\n" + "=" * 60)
    print("  Phase 5 — Full Training")
    print("=" * 60)

    # --- Fix F: Seed everything ---
    seed = cfg.get("experiment", {}).get("seed", 42)
    seed_everything(seed)

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
    accumulation_steps = t.get("accumulation_steps", 4)
    frozen_backbone_epochs = t.get("frozen_backbone_epochs", t.get("stage1_epochs", 5))
    min_finetune_epochs = t.get("min_finetune_epochs", 2)

    # Fix B: freeze BN stats throughout
    freeze_bn = t.get("freeze_backbone_bn_stats", True)

    # --- Stage 1: freeze backbone ---
    phase = 1
    optimizer, scheduler, scaler = _build_stage1(model, cfg)

    start_epoch = 1
    best_macro_auc = 0.0
    epochs_without_improvement = 0
    finetune_epochs_done = 0

    # --- Metric logging ---
    run_id = cfg.get("experiment", {}).get("run_id", "default")
    output_dir = cfg["paths"]["output_dir"]
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics_log.csv")

    # --- Resume from checkpoint (Fix F: phase-aware) ---
    if resume:
        ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        resumed_epoch = ckpt["epoch"]
        start_epoch = resumed_epoch + 1
        best_macro_auc = ckpt["best_macro_auc"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        phase = ckpt.get("phase", 1 if resumed_epoch <= frozen_backbone_epochs else 2)

        # Fix F: restore the ACTUAL phase
        if phase == 2:
            optimizer, scheduler, scaler = _build_stage2(model, cfg)
        # else: stage-1 objects already built above

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Fix F: restore RNG states
        if "rng_states" in ckpt:
            set_rng_states(ckpt["rng_states"])

        print(f"[train] Resumed from epoch {resumed_epoch} phase {phase} "
              f"(best_macro_auc={best_macro_auc:.4f})")
    else:
        # Fresh run — wipe metrics_log.csv so header is written cleanly
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs + 1):
        # Stage transition: unfreeze layer4 after frozen_backbone_epochs
        if phase == 1 and epoch == frozen_backbone_epochs + 1:
            print(f"\n[train] Stage 2: unfreezing layer4 (backbone BN stays frozen)")
            phase = 2
            optimizer, scheduler, scaler = _build_stage2(model, cfg, prev_optimizer=optimizer)
            finetune_epochs_done = 0

        print(f"\nEpoch {epoch}/{num_epochs}  [phase {phase}]")

        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp,
            accumulation_steps=accumulation_steps,
            freeze_bn=freeze_bn,
        )

        # Validate
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device, class_names, use_amp
        )

        # Scheduler step
        if isinstance(scheduler, WarmupThenPlateau):
            scheduler.step(val_metrics["macro_auc"])
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["macro_auc"])
        else:
            scheduler.step()

        # Track fine-tune epochs
        if phase == 2:
            finetune_epochs_done += 1

        # Log (Fix F: full-precision for selection, rounded for display)
        display = round_metrics(val_metrics)
        display["epoch"] = epoch
        display["train_loss"] = round(train_loss, 4)
        display["phase"] = phase
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={display['overall_accuracy']:.4f}  "
              f"macro_auc={display['macro_auc']:.4f}  "
              f"macro_f1={display['macro_f1']:.4f}")

        # Save metrics CSV
        write_header = not os.path.isfile(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(display.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(display)

        # Best model checkpoint (Fix F: full-precision comparison)
        if val_metrics["macro_auc"] > best_macro_auc:
            best_macro_auc = val_metrics["macro_auc"]
            epochs_without_improvement = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            _save_checkpoint(
                best_ckpt_path, epoch, phase, model, optimizer, scheduler,
                scaler, best_macro_auc, epochs_without_improvement, cfg,
            )
            print(f"  ★ New best macro-AUC: {best_macro_auc:.6f} — saved to {best_ckpt_path}")
        else:
            epochs_without_improvement += 1

        # Save resumable checkpoint (every epoch, atomic)
        last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
        _save_checkpoint(
            last_ckpt_path, epoch, phase, model, optimizer, scheduler,
            scaler, best_macro_auc, epochs_without_improvement, cfg,
        )
        print(f"  💾 Saved last_checkpoint.pt (epoch {epoch}, phase {phase})")

        # Fix F: Phase-aware early stopping
        if epochs_without_improvement >= patience:
            if phase == 2 and finetune_epochs_done < min_finetune_epochs:
                # Don't stop during the mandatory fine-tune minimum
                print(f"  (patience exhausted but only {finetune_epochs_done}/"
                      f"{min_finetune_epochs} fine-tune epochs done — continuing)")
            else:
                print(f"\n[train] Early stopping — no improvement for {patience} epochs.")
                break

    print(f"\n[train] Training complete. Best macro-AUC: {best_macro_auc:.6f}")
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
