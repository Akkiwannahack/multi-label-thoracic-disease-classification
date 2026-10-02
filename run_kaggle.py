"""
run_kaggle.py — Complete Kaggle Notebook Runner

This script runs all phases end-to-end on Kaggle. Copy this file and the
thoracic-xray/ directory to your Kaggle notebook, attach the NIH ChestX-ray14
dataset, and execute:

    !python run_kaggle.py

Or run individual phases:
    !python run_kaggle.py --phase 1      # Dataset preparation only
    !python run_kaggle.py --phase 3      # Model shape test only
    !python run_kaggle.py --phase 4      # Overfit sanity check
    !python run_kaggle.py --phase 5      # Full training
    !python run_kaggle.py --phase 6-9    # Explainability + reports
    !python run_kaggle.py --phase 10     # Evaluation
"""

import argparse
import json
import os
import sys
import time

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import yaml


def install_dependencies():
    """Install required packages (run once per Kaggle session)."""
    import subprocess

    print("=" * 60)
    print("  Installing dependencies...")
    print("=" * 60)

    packages = [
        "albumentations",
        "grad-cam",
        "einops",
        "fpdf2",
        "gradio",
    ]

    for pkg in packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=False,
        )

    # Try installing mamba-ssm (may fail on some Kaggle CUDA versions)
    print("\n[setup] Attempting mamba-ssm install (fallback to pure-PyTorch if it fails)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "causal-conv1d>=1.2.0", "--no-build-isolation"],
        check=False,
    )
    if result.returncode == 0:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "mamba-ssm", "--no-build-isolation"],
            check=False,
        )
        if result.returncode == 0:
            print("[setup] ✓ mamba-ssm installed successfully (CUDA kernel)")
        else:
            print("[setup] ⚠ mamba-ssm CUDA build failed — will use pure-PyTorch SSM fallback")
    else:
        print("[setup] ⚠ causal-conv1d failed — will use pure-PyTorch SSM fallback")

    print("[setup] ✓ Dependencies installed\n")


def load_config(config_path: str = None) -> dict:
    """Load and patch config for Kaggle paths."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "base.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Auto-detect Kaggle dataset path
    kaggle_data_path = "/kaggle/input/data"
    if os.path.isdir(kaggle_data_path):
        cfg["dataset"]["root_dir"] = kaggle_data_path
        print(f"[config] Using Kaggle dataset path: {kaggle_data_path}")
    else:
        print(f"[config] WARNING: Kaggle data path not found at {kaggle_data_path}")
        print(f"[config] Using config default: {cfg['dataset']['root_dir']}")

    # Set output paths relative to project root
    for key in ["manifest_dir", "checkpoint_dir", "output_dir", "report_dir"]:
        cfg["paths"][key] = os.path.join(PROJECT_ROOT, cfg["paths"][key])
        os.makedirs(cfg["paths"][key], exist_ok=True)

    return cfg


def run_phase_0(cfg: dict):
    """Phase 0: Verify setup."""
    import torch

    print("\n" + "=" * 60)
    print("  Phase 0 — Setup Verification")
    print("=" * 60)

    print(f"  Python:  {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Check dataset
    root = cfg["dataset"]["root_dir"]
    csv_path = os.path.join(root, cfg["dataset"]["csv_file"])
    if os.path.isfile(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  Dataset: ✓ {len(df)} rows in {csv_path}")
    else:
        print(f"  Dataset: ✗ {csv_path} NOT FOUND")

    # Check mamba-ssm
    try:
        from mamba_ssm import Mamba
        print("  Mamba:   ✓ CUDA kernel available")
    except ImportError:
        print("  Mamba:   ⚠ CUDA kernel not available — using pure-PyTorch fallback")

    print()


def run_phase_1(cfg: dict):
    """Phase 1: Build dataset manifests."""
    from data.build_manifest import build_manifests
    build_manifests(cfg)


def run_phase_2_visual_check(cfg: dict):
    """Phase 2: Visual check of enhancement pipeline."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.enhancement import enhance_image

    print("\n" + "=" * 60)
    print("  Phase 2 — Enhancement Visual Check")
    print("=" * 60)

    # Find sample images
    root = cfg["dataset"]["root_dir"]
    sample_images = []
    for sub in cfg["dataset"]["image_dirs"]:
        img_dir = os.path.join(root, sub)
        if os.path.isdir(img_dir):
            files = [f for f in os.listdir(img_dir) if f.endswith(".png")][:2]
            for f in files:
                sample_images.append(os.path.join(img_dir, f))
        if len(sample_images) >= 6:
            break

    if not sample_images:
        print("[phase2] No sample images found — skipping visual check")
        return

    fig, axes = plt.subplots(len(sample_images), 2, figsize=(10, 3 * len(sample_images)))
    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)

    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        enhanced = enhance_image(img, normalize=False,
                                  image_size=cfg["enhancement"]["image_size"])

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Original: {os.path.basename(img_path)}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(enhanced)
        axes[i, 1].set_title("Enhanced (CLAHE + Denoise + Sharpen)", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(cfg["paths"]["output_dir"], "enhancement_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[phase2] ✓ Enhancement comparison saved → {out_path}")


def run_phase_3(cfg: dict):
    """Phase 3: Model architecture test."""
    import torch
    from models.hybrid_model import build_model
    from models.heads import count_parameters, parameter_summary, verify_no_softmax, test_model_output_shape

    print("\n" + "=" * 60)
    print("  Phase 3 — Model Architecture Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # Parameter count
    total = count_parameters(model, only_trainable=False)
    trainable = count_parameters(model, only_trainable=True)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    for name, info in parameter_summary(model).items():
        print(f"    {name}: {info['total']:,} total, {info['trainable']:,} trainable")

    # No softmax check
    assert verify_no_softmax(model), "FAIL: Softmax found in multi-label model!"
    print("  ✓ No softmax in model")

    # Shape test
    assert test_model_output_shape(model, batch_size=2, device=str(device))

    # Forward + backward test
    dummy = torch.randn(4, 3, 224, 224, device=device)
    logits = model(dummy)
    loss = logits.sum()
    loss.backward()
    print(f"  ✓ Forward + backward pass OK on batch of 4")
    print(f"  ✓ Output shape: {logits.shape}")


def run_phase_4(cfg: dict):
    """Phase 4: Overfit sanity check."""
    import torch
    from training.train import overfit_sanity_check

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    passed = overfit_sanity_check(cfg, device)
    if not passed:
        print("\n⚠ Phase 4 FAILED — fix issues before proceeding to Phase 5!")
        return False
    return True


def run_phase_5(cfg: dict):
    """Phase 5: Full training."""
    import torch
    from training.train import train_full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_full(cfg, device)


def run_phase_6_7_8_9(cfg: dict):
    """Phases 6-9: Explainability + Reports on sample test images."""
    import torch
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from models.hybrid_model import build_model
    from explainability.gradcam_pp import generate_heatmap, overlay_heatmap
    from explainability.localization import localize
    from explainability.confidence import predict_with_confidence
    from reporting.generate_report import generate_report
    from data.enhancement import enhance_image

    print("\n" + "=" * 60)
    print("  Phases 6-9 — Explainability & Reports")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[phase6-9] Checkpoint not found at {ckpt_path} — skipping")
        return

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[phase6-9] Loaded model from epoch {ckpt.get('epoch', '?')}")

    # Get 5 sample test images
    import pandas as pd
    test_csv = os.path.join(cfg["paths"]["manifest_dir"], "manifest_test.csv")
    if not os.path.isfile(test_csv):
        print("[phase6-9] No test manifest — skipping")
        return

    test_df = pd.read_csv(test_csv)
    sample_df = test_df.head(5)

    class_names = cfg["dataset"]["class_names"]
    enh_cfg = cfg["enhancement"]
    gc_cfg = cfg["gradcam"]
    loc_cfg = cfg["localization"]

    report_dir = cfg["paths"]["report_dir"]
    os.makedirs(report_dir, exist_ok=True)

    for idx, row in sample_df.iterrows():
        img_path = row["image_path"]
        img_name = os.path.basename(img_path)
        print(f"\n  Processing: {img_name}")

        # Generate PDF report
        report_path = os.path.join(report_dir, f"report_{img_name.replace('.png', '.pdf')}")
        try:
            generate_report(
                img_path, model, cfg, report_path,
                patient_id=str(row.get("Patient ID", "Unknown")),
                device=device,
            )
            print(f"    ✓ Report saved → {report_path}")
        except Exception as e:
            print(f"    ✗ Report failed: {e}")

    # Generate heatmap grid for visual review
    print("\n[phase6-9] Generating heatmap overview grid...")
    try:
        fig, axes = plt.subplots(len(sample_df), 3, figsize=(12, 4 * len(sample_df)))
        if len(sample_df) == 1:
            axes = axes.reshape(1, -1)

        for i, (_, row) in enumerate(sample_df.iterrows()):
            img_path = row["image_path"]
            raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                continue

            enhanced_rgb = enhance_image(raw_img, normalize=False,
                                          image_size=enh_cfg["image_size"])
            enhanced_norm = enhance_image(raw_img, normalize=True,
                                           image_size=enh_cfg["image_size"])
            tensor = torch.from_numpy(enhanced_norm).permute(2, 0, 1).float()

            result = predict_with_confidence(
                model, tensor, class_names=class_names, mc_passes=10, device=device
            )

            axes[i, 0].imshow(raw_img, cmap="gray")
            axes[i, 0].set_title("Original", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(enhanced_rgb)
            axes[i, 1].set_title("Enhanced", fontsize=9)
            axes[i, 1].axis("off")

            if result["detected_diseases"]:
                top = result["detected_diseases"][0]
                heatmap = generate_heatmap(model, tensor, top["class_index"], device=device)
                overlay = overlay_heatmap(enhanced_rgb, heatmap, alpha=gc_cfg["alpha_blend"])
                regions = localize(heatmap, percentile=gc_cfg["heatmap_threshold_percentile"])
                title = f"Grad-CAM: {top['class_name']} ({top['probability']:.0%})\n{', '.join(regions)}"
            else:
                overlay = enhanced_rgb
                title = "No findings"

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(title, fontsize=8)
            axes[i, 2].axis("off")

        plt.tight_layout()
        grid_path = os.path.join(cfg["paths"]["output_dir"], "heatmap_overview.png")
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"  ✓ Heatmap overview → {grid_path}")
    except Exception as e:
        print(f"  ✗ Heatmap grid failed: {e}")


def run_phase_10(cfg: dict):
    """Phase 10: Evaluation & baseline comparison."""
    import torch
    from training.evaluate import evaluate_model, print_evaluation_report, save_comparison_table
    from models.hybrid_model import build_model
    from data.dataset import build_dataloaders

    print("\n" + "=" * 60)
    print("  Phase 10 — Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["dataset"]["class_names"]

    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[phase10] Checkpoint not found — skipping")
        return

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = build_dataloaders(cfg)
    if "test" not in loaders:
        print("[phase10] No test manifest — skipping")
        return

    metrics, _, _ = evaluate_model(model, loaders["test"], device, class_names)
    print_evaluation_report(metrics, class_names, "Hybrid ResNet50 + Vision Mamba")

    results = [{
        "model": "ResNet50 + Vision Mamba (Ours)",
        "macro_auc": metrics["macro_auc"],
        "micro_auc": metrics.get("micro_auc", ""),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
    }]

    output_path = os.path.join(cfg["paths"]["output_dir"], "baseline_comparison.csv")
    save_comparison_table(results, output_path)


def run_training_curves(cfg: dict):
    """Plot training curves from metrics_log.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_path = os.path.join(cfg["paths"]["output_dir"], "metrics_log.csv")
    if not os.path.isfile(metrics_path):
        print("[curves] No metrics log found — skipping")
        return

    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], "b-", label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(df["epoch"], df["macro_auc"], "g-", label="Macro AUC")
    if "micro_auc" in df.columns:
        axes[1].plot(df["epoch"], df["micro_auc"], "m-", label="Micro AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Validation AUC-ROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(df["epoch"], df["macro_f1"], "c-", label="Macro F1")
    axes[2].plot(df["epoch"], df["micro_f1"], "y-", label="Micro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation F1 Scores")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(cfg["paths"]["output_dir"], "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[curves] ✓ Training curves saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run thoracic disease classification pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase to run: 0, 1, 2, 3, 4, 5, 6-9, 10, curves, all")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip pip install step")
    args = parser.parse_args()

    if not args.skip_install and args.phase in ("all", "0"):
        install_dependencies()

    cfg = load_config(args.config)

    phases = args.phase.lower()

    if phases in ("all", "0"):
        run_phase_0(cfg)

    if phases in ("all", "1"):
        run_phase_1(cfg)

    if phases in ("all", "2"):
        run_phase_2_visual_check(cfg)

    if phases in ("all", "3"):
        run_phase_3(cfg)

    if phases in ("all", "4"):
        passed = run_phase_4(cfg)
        if phases == "all" and not passed:
            print("\n⚠ Stopping — Phase 4 overfit test failed!")
            sys.exit(1)

    if phases in ("all", "5"):
        run_phase_5(cfg)
        run_training_curves(cfg)

    if phases in ("all", "6-9"):
        run_phase_6_7_8_9(cfg)

    if phases in ("all", "10"):
        run_phase_10(cfg)

    if phases == "curves":
        run_training_curves(cfg)

    if phases == "all":
        print("\n" + "=" * 60)
        print("  ✓ ALL PHASES COMPLETE")
        print("=" * 60)
        print(f"  Outputs: {cfg['paths']['output_dir']}")
        print(f"  Reports: {cfg['paths']['report_dir']}")
        print(f"  Checkpoint: {os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pt')}")


if __name__ == "__main__":
    main()
