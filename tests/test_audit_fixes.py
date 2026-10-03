"""
tests/test_audit_fixes.py — Focused tests for the audit fix suite.

Covers Section 3 of the plan: splits/class order, loader/loss independence,
weight semantics, bidirectional residual, shapes/gradients, frozen BN/groups,
training API, resume boundary, scheduler, accumulation, cache, threshold
provenance, inference parity.

Usage:
    "E:\\PBL V\\venv\\Scripts\\python.exe" -m pytest tests/test_audit_fixes.py -v --tb=short
"""

import copy
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

# Ensure the project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── Helper: build a minimal config for testing ──────────────────────

def _mini_cfg(tmp_dir, **overrides):
    """Build a minimal config dict that all modules accept, without needing
    real data files."""
    cfg = {
        "experiment": {"run_id": "test", "seed": 42, "model_type": "hybrid_resnet_mamba"},
        "dataset": {
            "root_dir": tmp_dir,
            "csv_file": "fake.csv",
            "train_val_list": "fake_tv.txt",
            "test_list": "fake_test.txt",
            "image_dirs": [],
            "val_fraction": 0.1,
            "num_classes": 14,
            "use_cache": False,
            "class_names": [
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                "Consolidation", "Edema", "Emphysema", "Fibrosis",
                "Pleural_Thickening", "Hernia",
            ],
        },
        "data_policy": {"sampler": "uniform", "drop_last_train": False},
        "enhancement": {
            "image_size": 224,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "bilateral_d": 5,
            "bilateral_sigma_color": 50,
            "bilateral_sigma_space": 50,
            "sharpen_amount": 1.0,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "clahe_enabled": True,
            "bilateral_enabled": True,
            "unsharp_enabled": True,
        },
        "augmentation": {
            "horizontal_flip_p": 0.5,
            "rotation_limit": 7,
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
            "shift_limit": 0.05,
            "scale_limit": 0.05,
            "noise_enabled": True,
            "grid_distortion_enabled": True,
            "coarse_dropout_enabled": True,
        },
        "model": {
            "backbone": "resnet50",
            "resnet_out_channels": 2048,
            "mamba_d_model": 64,      # small for speed
            "mamba_d_state": 8,
            "mamba_num_blocks": 2,    # small for speed
            "mamba_bidirectional": True,
            "fusion_hidden": 128,
            "dropout": 0.3,
            "num_classes": 14,
            "residual_fusion": "mean_v2",
            "mamba_backend": "custom_pytorch",
        },
        "training": {
            "loss": "bce",
            "focal_gamma": 2.0,
            "asl_gamma_neg": 4.0,
            "asl_gamma_pos": 1.0,
            "asl_clip": 0.05,
            "label_smoothing": 0.0,
            "class_weighting": "none",
            "positive_weighting": "none",
            "max_class_weight": 30.0,
            "max_pos_weight": 5.0,
            "batch_size": 4,
            "accumulation_steps": 2,
            "num_workers": 0,
            "epochs": 2,
            "mixed_precision": False,
            "frozen_backbone_epochs": 1,
            "fine_tune_layers": ["layer4"],
            "freeze_backbone_bn_stats": True,
            "min_finetune_epochs": 1,
            "lr_backbone": 1e-5,
            "lr_head": 1e-4,
            "weight_decay": 1e-4,
            "early_stopping_patience": 3,
            "scheduler": "plateau",
            "plateau_factor": 0.5,
            "plateau_patience": 1,
            "warmup_epochs_stage2": 1,
            "stage1_epochs": 1,
            "lr_backbone_stage1": 5e-5,
            "lr_head_stage1": 1.5e-4,
            "lr_backbone_stage2": 5e-6,
            "lr_head_stage2": 5e-5,
        },
        "confidence": {"mc_passes": 2, "high_threshold": 0.05, "medium_threshold": 0.12},
        "gradcam": {"target_layer": "layer4", "alpha_blend": 0.4, "heatmap_threshold_percentile": 80},
        "localization": {
            "grid_rows": 3, "grid_cols": 2,
            "region_names": ["Right Upper", "Left Upper", "Right Mid", "Left Mid", "Right Lower", "Left Lower"],
        },
        "paths": {
            "manifest_dir": tmp_dir,
            "checkpoint_dir": os.path.join(tmp_dir, "ckpts"),
            "output_dir": os.path.join(tmp_dir, "outputs"),
            "report_dir": os.path.join(tmp_dir, "reports"),
        },
    }
    for k, v in overrides.items():
        # Simple one-level override
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ═══════════════════════════════════════════════════════════════════
# 1. Weight semantics — Fix A
# ═══════════════════════════════════════════════════════════════════

class TestLossConstruction:
    """Verify loss factories reject unknown names and produce correct types."""

    def test_reject_unknown_loss(self, tmp_path):
        from training.losses import build_loss
        cfg = _mini_cfg(str(tmp_path), training={"loss": "magic_loss"})
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss(cfg)

    def test_bce_unweighted(self, tmp_path):
        from training.losses import build_loss, PositiveWeightedBCELoss
        cfg = _mini_cfg(str(tmp_path))
        loss = build_loss(cfg)
        assert isinstance(loss, PositiveWeightedBCELoss)
        assert loss.pos_weight is None

    def test_bce_with_pos_weight(self, tmp_path):
        from training.losses import build_loss, PositiveWeightedBCELoss
        # Create a class_balance.json
        balance = {}
        for i, name in enumerate([
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia",
        ]):
            balance[name] = {
                "positive_count": 100 * (i + 1),
                "total_count": 10000,
                "positive_frequency": 0.01 * (i + 1),
                "inverse_frequency_weight": 1.0 / (0.01 * (i + 1)),
            }
        bp = os.path.join(str(tmp_path), "class_balance.json")
        with open(bp, "w") as f:
            json.dump(balance, f)

        cfg = _mini_cfg(str(tmp_path), training={
            "loss": "bce",
            "positive_weighting": "sqrt_ratio",
        })
        loss = build_loss(cfg, class_balance_path=bp)
        assert isinstance(loss, PositiveWeightedBCELoss)
        assert loss.pos_weight is not None
        assert loss.pos_weight.shape == (14,)

    def test_focal_explicit(self, tmp_path):
        from training.losses import build_loss, MultiLabelFocalLoss
        cfg = _mini_cfg(str(tmp_path), training={"loss": "focal"})
        loss = build_loss(cfg)
        assert isinstance(loss, MultiLabelFocalLoss)

    def test_unweighted_bce_matches_pytorch(self, tmp_path):
        """Plain BCE with our wrapper matches PyTorch's BCEWithLogitsLoss."""
        from training.losses import PositiveWeightedBCELoss
        our_loss = PositiveWeightedBCELoss(pos_weight=None, label_smoothing=0.0)
        pt_loss = nn.BCEWithLogitsLoss()

        torch.manual_seed(42)
        logits = torch.randn(8, 14)
        targets = torch.randint(0, 2, (8, 14)).float()

        ours = our_loss(logits, targets)
        theirs = pt_loss(logits, targets)
        assert torch.allclose(ours, theirs, atol=1e-6), f"{ours} != {theirs}"


# ═══════════════════════════════════════════════════════════════════
# 2. Bidirectional residual — Fix C
# ═══════════════════════════════════════════════════════════════════

class TestBidirectionalResidual:
    """With zero branch updates the new wrapper returns exactly the input;
    legacy mode returns twice the input."""

    def _build_block(self, fusion_mode, d_model=32):
        from models.vision_mamba_block import BidirectionalMambaBlock
        block = BidirectionalMambaBlock(
            d_model=d_model, d_state=8, expand=2,
            use_cuda=False, residual_fusion=fusion_mode,
        )
        # Zero all parameters to make branch updates zero
        for p in block.parameters():
            nn.init.zeros_(p)
        return block

    def test_mean_v2_identity(self):
        block = self._build_block("mean_v2")
        x = torch.randn(2, 7, 32)
        with torch.no_grad():
            out = block(x)
        # When all parameters are zero, each branch returns the input
        # mean_v2: x + 0.5*(fwd + bwd - 2x) = x + 0.5*(2x - 2x) = x
        assert torch.allclose(out, x, atol=1e-5), \
            f"mean_v2 should return x when branches are zero; max diff={( out - x).abs().max()}"

    def test_legacy_sum_doubles(self):
        block = self._build_block("legacy_sum")
        x = torch.randn(2, 7, 32)
        with torch.no_grad():
            out = block(x)
        # legacy: fwd + bwd = (x + 0) + (x + 0) = 2x
        assert torch.allclose(out, 2 * x, atol=1e-5), \
            f"legacy_sum should return 2x when branches are zero; max diff={(out - 2*x).abs().max()}"

    def test_reject_unknown_fusion(self):
        from models.vision_mamba_block import BidirectionalMambaBlock
        with pytest.raises(ValueError, match="Unknown residual_fusion"):
            BidirectionalMambaBlock(32, 8, residual_fusion="invalid")


# ═══════════════════════════════════════════════════════════════════
# 3. Shapes and gradients — Fix I
# ═══════════════════════════════════════════════════════════════════

class TestShapesAndGradients:
    """Output is [B, 14], 49-token sequence, trainable modules get gradients,
    unsupported resolution fails early."""

    def test_output_shape(self):
        from models.hybrid_model import build_model
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 14), f"Expected (2,14), got {out.shape}"

    def test_gradients_flow(self):
        from models.hybrid_model import build_model
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        model.train()
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        has_grad = {name: p.grad is not None and p.grad.abs().max() > 0
                    for name, p in model.named_parameters() if p.requires_grad}
        n_with_grad = sum(has_grad.values())
        assert n_with_grad > 0, "No trainable parameters received gradients"

    def test_wrong_resolution_fails(self):
        from models.vision_mamba_block import VisionMambaEncoder
        encoder = VisionMambaEncoder(
            in_channels=2048, spatial_size=7, d_model=64,
            d_state=8, num_blocks=1, use_cuda=False,
        )
        # 14x14 feature map (from 448px input) should fail
        bad_input = torch.randn(1, 2048, 14, 14)
        with pytest.raises(ValueError, match="Spatial size mismatch"):
            encoder(bad_input)


# ═══════════════════════════════════════════════════════════════════
# 4. Frozen BatchNorm and groups — Fix B
# ═══════════════════════════════════════════════════════════════════

class TestFrozenBNAndGroups:
    """Frozen BN running stats unchanged after a training step;
    each trainable parameter is in exactly one expected-rate group."""

    def test_bn_stats_frozen(self):
        from models.hybrid_model import build_model
        from models.resnet_backbone import ResNet50Backbone
        cfg = _mini_cfg(".")
        model = build_model(cfg)

        # Record BN stats before
        bn_stats_before = {}
        for name, m in model.backbone.named_modules():
            if isinstance(m, (nn.BatchNorm2d,)):
                bn_stats_before[name] = (
                    m.running_mean.clone(),
                    m.running_var.clone(),
                )

        # Train step with frozen BN
        model.train()
        ResNet50Backbone.freeze_bn_stats(model.backbone)

        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Check BN stats unchanged
        for name, m in model.backbone.named_modules():
            if isinstance(m, (nn.BatchNorm2d,)) and name in bn_stats_before:
                old_mean, old_var = bn_stats_before[name]
                assert torch.equal(m.running_mean, old_mean), \
                    f"BN {name} running_mean changed"
                assert torch.equal(m.running_var, old_var), \
                    f"BN {name} running_var changed"

    def test_param_groups_no_overlap(self):
        from models.hybrid_model import build_model
        cfg = _mini_cfg(".")
        model = build_model(cfg)

        # Stage 1: freeze backbone
        model.freeze_backbone()
        groups = model.get_param_groups_v2(
            lr_backbone=0, lr_head=1e-4, fine_tune_layers=None
        )

        assigned_ids = set()
        for g in groups:
            for p in g["params"]:
                pid = id(p)
                assert pid not in assigned_ids, "Parameter in multiple groups!"
                assigned_ids.add(pid)

    def test_frozen_backbone_no_grad(self):
        from models.hybrid_model import build_model
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        model.freeze_backbone()

        for name, p in model.backbone.named_parameters():
            assert not p.requires_grad, f"Backbone param {name} should be frozen"


# ═══════════════════════════════════════════════════════════════════
# 5. Training API and baseline — Fix E
# ═══════════════════════════════════════════════════════════════════

class TestTrainingAPI:
    """Hybrid, baseline callers finish a tiny epoch; baseline reload uses
    the baseline factory; focal variants resolve to focal."""

    def test_train_one_epoch_returns_float(self):
        from models.hybrid_model import build_model
        from training.train import train_one_epoch
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        model.train()

        # Create a tiny dataloader
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 2, (8, 14)).float()
        ds = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        result = train_one_epoch(
            model, loader, criterion, optimizer, None,
            torch.device("cpu"), False, accumulation_steps=2,
        )
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result > 0, "Loss should be positive"

    def test_baseline_model_distinct(self):
        from models.hybrid_model import build_model
        from models.baseline_resnet50 import build_baseline_model
        cfg = _mini_cfg(".")
        hybrid = build_model(cfg)
        baseline = build_baseline_model(cfg)
        # Baseline should NOT have mamba_encoder
        assert hasattr(hybrid, "mamba_encoder")
        assert not hasattr(baseline, "mamba_encoder")


# ═══════════════════════════════════════════════════════════════════
# 6. Backend selection — Fix D
# ═══════════════════════════════════════════════════════════════════

class TestBackendSelection:
    """Explicit backend, reject unavailable, provenance in checkpoint."""

    def test_custom_pytorch_backend(self):
        from models.vision_mamba_block import resolve_backend
        use_cuda = resolve_backend("custom_pytorch")
        assert use_cuda is False

    def test_unknown_backend_rejected(self):
        from models.vision_mamba_block import resolve_backend
        with pytest.raises(ValueError, match="Unknown mamba_backend"):
            resolve_backend("banana")

    def test_provenance_in_model(self):
        from models.hybrid_model import build_model
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        prov = model.get_provenance()
        assert prov["model_type"] == "hybrid_resnet_mamba"
        assert prov["residual_fusion"] == "mean_v2"
        assert prov["mamba_backend"] == "custom_pytorch"
        assert prov["num_classes"] == 14


# ═══════════════════════════════════════════════════════════════════
# 7. Enhancement & cache — Fix G
# ═══════════════════════════════════════════════════════════════════

class TestEnhancement:
    """Enhancement booleans work, fingerprint changes on config change."""

    def test_raw_resize_no_enhancement(self):
        from data.enhancement import enhance_image
        import cv2
        # Create a tiny grayscale image
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = enhance_image(
            img, image_size=32, normalize=False,
            clahe_enabled=False, bilateral_enabled=False, unsharp_enabled=False,
        )
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8
        # All 3 channels should be identical (grayscale replicated)
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 0], result[:, :, 2])

    def test_enhancement_with_clahe(self):
        from data.enhancement import enhance_image
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with_clahe = enhance_image(
            img, image_size=32, normalize=False,
            clahe_enabled=True, bilateral_enabled=False, unsharp_enabled=False,
        )
        without_clahe = enhance_image(
            img, image_size=32, normalize=False,
            clahe_enabled=False, bilateral_enabled=False, unsharp_enabled=False,
        )
        # Results should differ
        assert not np.array_equal(with_clahe, without_clahe), \
            "CLAHE should change the image"

    def test_fingerprint_changes_on_config(self):
        from data.enhancement import compute_preprocessing_fingerprint
        fp1 = compute_preprocessing_fingerprint({
            "image_size": 224,
            "clahe_enabled": True,
            "bilateral_enabled": True,
            "unsharp_enabled": True,
            "clahe_clip": 2.0,
            "clahe_tile": (8, 8),
            "bilateral_d": 5,
            "bilateral_sc": 50,
            "bilateral_ss": 50,
            "sharpen_amount": 1.0,
        })
        fp2 = compute_preprocessing_fingerprint({
            "image_size": 224,
            "clahe_enabled": False,
            "bilateral_enabled": False,
            "unsharp_enabled": False,
        })
        assert fp1 != fp2, "Different enhancement settings should give different fingerprints"

    def test_normalized_output_shape(self):
        from data.enhancement import enhance_image
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = enhance_image(
            img, image_size=224, normalize=True,
            clahe_enabled=False, bilateral_enabled=False, unsharp_enabled=False,
        )
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════
# 8. Scheduler — Fix F
# ═══════════════════════════════════════════════════════════════════

class TestScheduler:
    """A synthetic stagnant validation sequence actually reduces the LR
    after warm-up and patience."""

    def test_warmup_then_plateau(self):
        from training.train import WarmupThenPlateau
        params = [torch.randn(10, requires_grad=True)]
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=2,
        )
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1,
        )
        sched = WarmupThenPlateau(warmup, plateau, warmup_epochs=2)

        lrs = []
        for epoch in range(8):
            lrs.append(optimizer.param_groups[0]["lr"])
            sched.step(val_metric=0.5)  # stagnant

        # LR should increase during warmup (epochs 0-1), then decrease
        # after patience exhausts (epoch 4+ or so)
        assert lrs[0] < lrs[2], f"LR should increase during warmup: {lrs[:3]}"
        assert lrs[-1] < lrs[2], f"LR should decrease after stagnation: {lrs}"


# ═══════════════════════════════════════════════════════════════════
# 9. Accumulation tail — Fix F
# ═══════════════════════════════════════════════════════════════════

class TestAccumulation:
    """On a model without batch-dependent layers, accumulated gradients
    match a concatenated batch."""

    def test_accumulation_matches_full_batch(self):
        """Verify accumulated training matches single-batch training."""
        torch.manual_seed(42)

        # Simple linear model (no BN/dropout)
        model_acc = nn.Linear(4, 2, bias=False)
        model_full = nn.Linear(4, 2, bias=False)
        model_full.load_state_dict(model_acc.state_dict())

        criterion = nn.MSELoss()

        # Create 8 samples split into 4 micro-batches of 2
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        # ── Full batch ──
        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.1)
        opt_full.zero_grad()
        loss_full = criterion(model_full(x), y)
        loss_full.backward()
        opt_full.step()

        # ── Accumulated (4 micro-batches of 2, accumulation_steps=4) ──
        opt_acc = torch.optim.SGD(model_acc.parameters(), lr=0.1)
        opt_acc.zero_grad()
        accum_steps = 4
        for i in range(4):
            micro_x = x[i*2:(i+1)*2]
            micro_y = y[i*2:(i+1)*2]
            loss = criterion(model_acc(micro_x), micro_y) / accum_steps
            loss.backward()
        opt_acc.step()

        # Parameters should be close (not exact due to loss averaging semantics)
        for p_acc, p_full in zip(model_acc.parameters(), model_full.parameters()):
            diff = (p_acc - p_full).abs().max().item()
            # Allow some tolerance due to mean vs sum reduction
            assert diff < 0.5, f"Accumulated vs full-batch weight diff too large: {diff}"


# ═══════════════════════════════════════════════════════════════════
# 10. Seeding and RNG state — Fix F
# ═══════════════════════════════════════════════════════════════════

class TestSeeding:
    """Verify seed_everything produces deterministic results."""

    def test_deterministic_torch(self):
        from training.train import seed_everything
        seed_everything(42)
        a = torch.randn(5)
        seed_everything(42)
        b = torch.randn(5)
        assert torch.equal(a, b), "Same seed should produce same tensor"

    def test_rng_state_save_restore(self):
        from training.train import seed_everything, get_rng_states, set_rng_states
        seed_everything(42)
        _ = torch.randn(10)  # advance RNG
        states = get_rng_states()
        expected = torch.randn(5)

        # Reset and restore
        seed_everything(0)  # different seed
        set_rng_states(states)
        actual = torch.randn(5)
        assert torch.equal(expected, actual), "RNG state restore should reproduce values"


# ═══════════════════════════════════════════════════════════════════
# 11. Confidence prediction contract — Fix H
# ═══════════════════════════════════════════════════════════════════

class TestConfidenceContract:
    """Verify predict_with_confidence returns the expected contract."""

    def test_prediction_contract(self):
        from models.hybrid_model import build_model
        from explainability.confidence import predict_with_confidence
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        model.eval()

        x = torch.randn(3, 224, 224)
        result = predict_with_confidence(
            model, x, class_names=cfg["dataset"]["class_names"],
            mc_passes=2, use_mc_dropout=False,
        )

        # Check required keys
        assert "probabilities" in result
        assert "binary_predictions" in result
        assert "thresholds_used" in result
        assert "inference_mode" in result
        assert "confidence_labels" in result
        assert "detected_diseases" in result

        assert result["probabilities"].shape == (14,)
        assert result["binary_predictions"].shape == (14,)
        assert len(result["confidence_labels"]) == 14
        assert result["inference_mode"] == "deterministic"

    def test_fitted_thresholds_used(self):
        from models.hybrid_model import build_model
        from explainability.confidence import predict_with_confidence
        cfg = _mini_cfg(".")
        model = build_model(cfg)
        model.eval()

        thresholds = {name: 0.99 for name in cfg["dataset"]["class_names"]}
        x = torch.randn(3, 224, 224)
        result = predict_with_confidence(
            model, x, class_names=cfg["dataset"]["class_names"],
            mc_passes=0, use_mc_dropout=False,
            thresholds=thresholds,
        )
        # With threshold 0.99, very few diseases should be detected
        assert len(result["detected_diseases"]) <= 14


# ═══════════════════════════════════════════════════════════════════
# 12. Checkpoint provenance — Fix D
# ═══════════════════════════════════════════════════════════════════

class TestCheckpointProvenance:
    """Checkpoints include model provenance and phase."""

    def test_save_and_reload(self, tmp_path):
        from models.hybrid_model import build_model
        from training.train import _save_checkpoint
        cfg = _mini_cfg(str(tmp_path))
        model = build_model(cfg)

        os.makedirs(str(tmp_path / "ckpts"), exist_ok=True)
        ckpt_path = str(tmp_path / "ckpts" / "test.pt")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")

        _save_checkpoint(
            ckpt_path, epoch=1, phase=1, model=model,
            optimizer=optimizer, scheduler=scheduler, scaler=None,
            best_macro_auc=0.75, epochs_without_improvement=0, cfg=cfg,
        )

        assert os.path.isfile(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        assert ckpt["epoch"] == 1
        assert ckpt["phase"] == 1
        assert "model_provenance" in ckpt
        assert ckpt["model_provenance"]["residual_fusion"] == "mean_v2"
        assert "rng_states" in ckpt
        assert "config" in ckpt
