"""
smoke_test.py — Section 4: One allowed GPU run

At most 20 batches forward/backward + one save/reload using recipe R.
Records: device, backend, precision, peak allocated/reserved GPU memory,
throughput, and verifies checkpoint roundtrip.

Usage:
    "E:\\PBL V\\venv\\Scripts\\python.exe" tests/smoke_test.py
"""

import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.hybrid_model import build_model
from models.resnet_backbone import ResNet50Backbone
from training.losses import build_loss
from training.train import (
    train_one_epoch, validate, seed_everything,
    _save_checkpoint, get_rng_states,
)
from data.dataset import ChestXrayDataset, build_dataloaders


def smoke_test():
    # Load recipe R config
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "recipe_R.yaml")
    if not os.path.isfile(cfg_path):
        print("[smoke] recipe_R.yaml not found — using base.yaml with overrides")
        cfg_path = os.path.join(PROJECT_ROOT, "configs", "base.yaml")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Override for smoke test
    cfg["training"]["epochs"] = 1
    cfg["training"]["mixed_precision"] = True
    cfg["dataset"]["use_cache"] = False  # Don't need cache for 20 batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] Device: {device}")
    if device.type == "cuda":
        print(f"[smoke] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[smoke] CUDA version: {torch.version.cuda}")
        torch.cuda.reset_peak_memory_stats()

    # Seed
    seed_everything(42)

    # Build model
    model = build_model(cfg).to(device)
    prov = model.get_provenance()
    print(f"[smoke] Model: {prov['model_type']}")
    print(f"[smoke] Backend: {prov['mamba_backend']}")
    print(f"[smoke] Fusion: {prov['residual_fusion']}")
    print(f"[smoke] d_model={prov['mamba_d_model']}, blocks={prov['mamba_num_blocks']}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] Total params: {total_params:,}")
    print(f"[smoke] Trainable params: {trainable_params:,}")

    # Stage 1: freeze backbone
    model.freeze_backbone()
    trainable_s1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] Trainable after freeze_backbone: {trainable_s1:,}")

    # Build loss
    balance_path = os.path.join(cfg["paths"]["manifest_dir"], "class_balance.json")
    criterion = build_loss(
        cfg,
        class_balance_path=balance_path if os.path.isfile(balance_path) else None,
    ).to(device)
    print(f"[smoke] Loss: {type(criterion).__name__}")

    # Build data loaders (real data)
    try:
        loaders = build_dataloaders(cfg)
        has_real_data = "train" in loaders and "val" in loaders
    except Exception as e:
        print(f"[smoke] Cannot build real dataloaders: {e}")
        has_real_data = False

    if not has_real_data:
        print("[smoke] No real data — using synthetic tensors for smoke test")
        images = torch.randn(40, 3, 224, 224)
        labels = torch.randint(0, 2, (40, 14)).float()
        ds = torch.utils.data.TensorDataset(images, labels)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(ds, batch_size=cfg["training"]["batch_size"])
    else:
        train_loader = loaders["train"]
        val_loader = loaders["val"]

    # Optimizer
    lr_head = cfg["training"].get("lr_head", 1e-4)
    param_groups = model.get_param_groups_v2(
        lr_backbone=0, lr_head=lr_head, fine_tune_layers=None
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["training"]["weight_decay"])

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    from torch.amp import GradScaler
    scaler = GradScaler(device="cuda") if use_amp else None
    accumulation_steps = cfg["training"].get("accumulation_steps", 4)

    print(f"\n[smoke] === Forward/Backward: up to 20 batches ===")
    print(f"[smoke] Batch size: {cfg['training']['batch_size']}")
    print(f"[smoke] Accumulation: {accumulation_steps}")
    print(f"[smoke] AMP: {use_amp}")

    # ── Train up to 20 batches ──
    model.train()
    ResNet50Backbone.freeze_bn_stats(model.backbone)

    batch_times = []
    batch_count = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    t_start = time.perf_counter()

    for images, labels in train_loader:
        if batch_count >= 20:
            break

        t_batch = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            from torch.amp import autocast
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_count + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(images)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

            if (batch_count + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        batch_count += 1
        elapsed_batch = time.perf_counter() - t_batch
        batch_times.append(elapsed_batch)

        if batch_count <= 5 or batch_count % 5 == 0:
            print(f"  batch {batch_count:2d}  loss={loss.item() * accumulation_steps:.4f}  "
                  f"time={elapsed_batch:.2f}s")

    t_total = time.perf_counter() - t_start
    avg_loss = running_loss / max(batch_count, 1)
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    samples_per_sec = cfg["training"]["batch_size"] / avg_batch_time if avg_batch_time > 0 else 0

    print(f"\n[smoke] Completed {batch_count} batches in {t_total:.1f}s")
    print(f"[smoke] Avg loss: {avg_loss:.4f}")
    print(f"[smoke] Avg batch time: {avg_batch_time*1000:.0f}ms")
    print(f"[smoke] Throughput: {samples_per_sec:.1f} samples/sec")

    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"[smoke] Peak GPU allocated: {peak_alloc:.2f} GB")
        print(f"[smoke] Peak GPU reserved:  {peak_reserved:.2f} GB")

    # ── Save/reload checkpoint ──
    print(f"\n[smoke] === Checkpoint save/reload ===")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    tmp_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmp_dir, "smoke_test.pt")

    _save_checkpoint(
        ckpt_path, epoch=1, phase=1, model=model,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        best_macro_auc=0.0, epochs_without_improvement=0, cfg=cfg,
    )
    ckpt_size = os.path.getsize(ckpt_path) / 1e6
    print(f"[smoke] Saved checkpoint: {ckpt_size:.1f} MB")

    # Reload
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2 = build_model(cfg).to(device)
    model2.load_state_dict(ckpt["model_state_dict"])

    # Verify provenance
    assert "model_provenance" in ckpt, "Missing model_provenance in checkpoint"
    assert ckpt["model_provenance"]["residual_fusion"] == "mean_v2"
    assert "rng_states" in ckpt, "Missing rng_states in checkpoint"
    assert ckpt["phase"] == 1, f"Expected phase 1, got {ckpt['phase']}"
    print(f"[smoke] Checkpoint roundtrip: ✓ provenance, ✓ rng_states, ✓ phase")

    # Verify identical predictions
    model.eval()
    model2.eval()
    test_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out1 = model(test_input)
        out2 = model2(test_input)
    max_diff = (out1 - out2).abs().max().item()
    print(f"[smoke] Prediction diff after reload: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Checkpoint reload changed predictions: diff={max_diff}"
    print(f"[smoke] Checkpoint roundtrip: ✓ identical predictions")

    # Cleanup
    os.remove(ckpt_path)
    os.rmdir(tmp_dir)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Device:           {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Backend:          {prov['mamba_backend']}")
    print(f"  Precision:        {'AMP (fp16)' if use_amp else 'fp32'}")
    print(f"  Batch size:       {cfg['training']['batch_size']} (accum {accumulation_steps})")
    print(f"  Batches run:      {batch_count}")
    print(f"  Avg loss:         {avg_loss:.4f}")
    print(f"  Avg batch time:   {avg_batch_time*1000:.0f}ms")
    print(f"  Throughput:       {samples_per_sec:.1f} samples/sec")
    if device.type == "cuda":
        print(f"  Peak GPU alloc:   {peak_alloc:.2f} GB")
        print(f"  Peak GPU reserved:{peak_reserved:.2f} GB")
    print(f"  Checkpoint size:  {ckpt_size:.1f} MB")
    print(f"  Roundtrip:        PASS")
    print(f"{'='*60}")
    print(f"  This is a SMOKE TEST, not a training result.")
    print(f"{'='*60}")


if __name__ == "__main__":
    smoke_test()
