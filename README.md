# 🫁 Multi-Label Thoracic Disease Classification

### ResNet50 × Vision Mamba Hybrid — NIH ChestX-ray14

A deep learning system for automated multi-label classification of **14 thoracic diseases** from chest X-ray images, combining the local feature extraction of **ResNet50** with the global sequence modeling of **Vision Mamba (Selective State-Space Models)**. The system provides Grad-CAM++ explainability heatmaps, MC Dropout confidence estimation, anatomical localization, and PDF diagnostic report generation.

> ⚠️ **Disclaimer:** This is a research prototype for academic and decision-support purposes only. It is **NOT** a clinical diagnostic tool.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Diseases Detected](#-diseases-detected)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
  - [Data Preparation](#1-data-preparation)
  - [Training](#2-training)
  - [Evaluation](#3-evaluation)
  - [Threshold Optimization](#4-threshold-optimization)
  - [Ablation Study](#5-ablation-study)
  - [Report Generation](#6-report-generation)
  - [Demo App](#7-demo-app)
- [Results](#-results)
- [Technical Details](#-technical-details)
- [Configuration](#-configuration)
- [License](#-license)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Hybrid Architecture** | ResNet50 backbone + Vision Mamba (bidirectional SSM) for combined local + global feature modeling |
| **Multi-Label Classification** | Simultaneous detection of 14 thoracic diseases from a single X-ray |
| **Image Enhancement** | CLAHE → Bilateral Denoising → Unsharp-Mask Sharpening pipeline |
| **Grad-CAM++ Explainability** | Visual heatmaps showing which image regions drive each prediction |
| **MC Dropout Confidence** | Uncertainty quantification via Monte Carlo Dropout (High / Medium / Low) |
| **Anatomical Localization** | Maps Grad-CAM++ activations to a 3×2 anatomical grid (Right/Left Upper/Mid/Lower) |
| **Per-Class Threshold Optimization** | F1-maximizing thresholds tuned per disease on the validation set |
| **PDF Diagnostic Reports** | Auto-generated reports with findings, heatmaps, confidence, and recommendations |
| **Gradio Demo App** | Interactive web interface for real-time X-ray analysis |
| **Ablation Study** | Systematic evaluation of individual component contributions |

---

## 🏗 Architecture

```
Input (B, 3, 224, 224)
    │
    ▼
ResNet50 (ImageNet-pretrained) → feature map (B, 2048, 7, 7)
    │                                       │
    ▼                                       ▼
Global Average Pool → (B, 2048)    Vision Mamba Encoder → (B, 256)
    │                                       │
    │   ┌───────────── concatenate ─────────┘
    │   │
    ▼   ▼
  (B, 2304) → LayerNorm → Linear(2304, 512) → GELU
              → Dropout(0.4) → Linear(512, 256) → GELU
              → Dropout(0.2) → Linear(256, 14) → logits
```

### Vision Mamba Encoder

The Vision Mamba branch converts the ResNet50 feature map into a **49-token sequence** (7×7 spatial grid), projects it to a working dimension, adds learnable 2D positional embeddings, and processes it through a stack of **bidirectional Mamba blocks**. Each block applies forward + backward selective state-space scans to capture context from all spatial directions — critical since images have no natural left-to-right causality.

Two backends are supported:
- **`mamba-ssm`** (compiled CUDA kernel) — fast, requires CUDA build
- **Pure-PyTorch fallback** — numerically equivalent, works everywhere

### Dual-Branch Fusion

| Branch | What it captures | Output |
|---|---|---|
| **GAP (ResNet50)** | Local textures, edges, fine-grained patterns | `(B, 2048)` |
| **Vision Mamba** | Long-range spatial dependencies, global context | `(B, 256)` |

The concatenated `(B, 2304)` vector passes through a fusion classification head that produces **14 independent logits** (no softmax — this is multi-label).

---

## 🩺 Diseases Detected

| # | Disease | # | Disease |
|---|---|---|---|
| 0 | Atelectasis | 7 | Pneumothorax |
| 1 | Cardiomegaly | 8 | Consolidation |
| 2 | Effusion | 9 | Edema |
| 3 | Infiltration | 10 | Emphysema |
| 4 | Mass | 11 | Fibrosis |
| 5 | Nodule | 12 | Pleural Thickening |
| 6 | Pneumonia | 13 | Hernia |

---

## 📁 Project Structure

```
thoracic-xray/
├── configs/
│   └── base.yaml                    # All hyperparameters and paths
│
├── data/
│   ├── augmentation.py              # Albumentations train/val pipelines
│   ├── build_cache.py               # Memmap cache builder for fast I/O
│   ├── build_manifest.py            # Train/val/test manifest CSV generator
│   ├── dataset.py                   # PyTorch Dataset + DataLoader factory
│   ├── enhancement.py               # CLAHE → Denoise → Sharpen pipeline
│   └── verify_cache.py              # Cache integrity checker
│
├── models/
│   ├── baseline_resnet50.py         # ResNet50-only baseline (for ablation)
│   ├── heads.py                     # Diagnostic utilities (param counting, shape tests)
│   ├── hybrid_model.py              # HybridResNetMamba (main model)
│   ├── resnet_backbone.py           # ResNet50 feature extractor (layer4 output)
│   └── vision_mamba_block.py        # Selective SSM blocks (CUDA + PyTorch backends)
│
├── training/
│   ├── ablation_study.py            # 6-variant ablation comparison
│   ├── calibration_mc_dropout.py    # MC Dropout inference + confidence buckets
│   ├── diagnose_instability.py      # Training instability diagnostics
│   ├── evaluate.py                  # Test-set evaluation + ROC curves
│   ├── losses.py                    # ASL, Focal Loss, Weighted BCE
│   ├── optimize_thresholds.py       # Per-class F1-maximizing thresholds
│   ├── train.py                     # Full training loop (staged fine-tuning)
│   └── train_baseline.py            # Baseline-only training script
│
├── explainability/
│   ├── confidence.py                # High-level MC Dropout prediction API
│   ├── gradcam_pp.py                # Grad-CAM++ / Score-CAM heatmap generation
│   └── localization.py              # Heatmap → anatomical region mapping
│
├── reporting/
│   └── generate_report.py           # PDF diagnostic report generator
│
├── app/
│   └── demo_app.py                  # Gradio web demo
│
├── outputs/                         # Training logs, metrics, plots, reports
├── checkpoints/                     # Model checkpoints (best + resumable)
├── requirements.txt
└── README.md
```

---

## ⚙ Setup & Installation

### Prerequisites

- Python ≥ 3.11
- CUDA-capable GPU (recommended; CPU works but is slow)
- ~45 GB disk space for the full NIH ChestX-ray14 dataset

### Installation

```bash
# Clone the repository
git clone https://github.com/Akkiwannahack/multi-label-thoracic-disease-classification.git
cd multi-label-thoracic-disease-classification/thoracic-xray

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch` ≥ 2.1 | Deep learning framework |
| `torchvision` | ResNet50 pretrained weights |
| `einops` | Tensor rearrangement for Vision Mamba |
| `albumentations` | Data augmentation |
| `opencv-python` | Image processing (CLAHE, bilateral filter) |
| `scikit-learn` | Metrics (AUC, F1, precision, recall) |
| `grad-cam` | Grad-CAM++ / Score-CAM |
| `fpdf2` | PDF report generation |
| `gradio` | Demo web application |
| `pandas` / `numpy` / `matplotlib` | Data handling and visualization |
| `pyyaml` | Configuration parsing |
| `tqdm` | Progress bars |
| `timm` | Model utilities |

**Optional:** Install [`mamba-ssm`](https://github.com/state-spaces/mamba) for faster CUDA-accelerated SSM kernels. The system automatically falls back to a pure-PyTorch implementation if unavailable.

---

## 📊 Dataset

This project uses the **NIH ChestX-ray14** dataset:

> *Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. IEEE CVPR, 2017.*

- **112,120** frontal-view chest X-rays from **30,805** unique patients
- **14** disease labels + "No Finding"
- Multi-label: each image can have **multiple** diseases
- Severely imbalanced (e.g., Hernia ≈ 0.2% vs. Infiltration ≈ 17%)

### Download

1. Download from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) or [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Place the 12 image folders (`images_001/` through `images_012/`) and metadata files under the configured `dataset.root_dir` path
3. Update `configs/base.yaml` with your local paths

---

## 🚀 Usage

All commands are run from the `thoracic-xray/` directory.

### 1. Data Preparation

```bash
# Build train/val/test manifest CSVs (patient-level split)
python -m data.build_manifest --config configs/base.yaml

# (Optional) Build memmap cache for faster data loading
python -m data.build_cache --config configs/base.yaml

# (Optional) Verify cache integrity
python -m data.verify_cache --config configs/base.yaml
```

### 2. Training

```bash
# Phase 4: Overfit sanity check (catches label/loss bugs early)
python -m training.train --config configs/base.yaml --overfit

# Phase 5: Full training (staged fine-tuning + early stopping)
python -m training.train --config configs/base.yaml

# Resume from checkpoint
python -m training.train --config configs/base.yaml --resume
```

**Training strategy:**
- **Stage 1** (epochs 1–5): Freeze ResNet50 `layer1`/`layer2`, train `layer3`/`layer4` + Mamba + classifier with differential LRs
- **Stage 2** (epochs 6–20): Unfreeze all layers with linear warmup → cosine annealing with warm restarts
- **Mixed precision** (FP16) with gradient accumulation (effective batch size = 64)
- **Early stopping** on validation macro-AUC (patience = 5)

### 3. Evaluation

```bash
# Evaluate on held-out test set
python -m training.evaluate --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt

# Compare against ResNet50-only baseline
python -m training.evaluate --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt \
    --baseline-checkpoint checkpoints/baseline_best.pt
```

### 4. Threshold Optimization

```bash
# Optimize per-class thresholds on validation set, evaluate on test set
python -m training.optimize_thresholds --config configs/base.yaml
```

### 5. Ablation Study

```bash
# Run all 6 variants
python -m training.ablation_study --config configs/base.yaml

# Run specific variants (a-f)
python -m training.ablation_study --config configs/base.yaml --variants a,b,e,f
```

### 6. Report Generation

```bash
# Generate a PDF diagnostic report for a single X-ray
python -m reporting.generate_report --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/xray.png \
    --output outputs/reports/report.pdf \
    --patient-id "Patient_001"
```

### 7. Demo App

```bash
# Launch interactive Gradio web demo
python -m app.demo_app --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt \
    --port 7860

# Create a shareable public link
python -m app.demo_app --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt --share
```

---

## 📈 Results

### Test Set Performance

| Metric | Value |
|---|---|
| **Macro AUC-ROC** | **0.7635** |
| **Micro AUC-ROC** | **0.8297** |
| **Overall Accuracy** | 0.7462 |
| **Macro F1** (default 0.5) | 0.2514 |
| **Macro F1** (optimized thresholds) | **0.2711** |

### Per-Class AUC Highlights

| Disease | F1 (default) | F1 (optimized) | Optimized Threshold |
|---|---|---|---|
| Effusion | 0.4134 | **0.4738** | 0.63 |
| Infiltration | 0.3990 | **0.4580** | 0.57 |
| Pneumothorax | 0.4087 | **0.4409** | 0.64 |
| Emphysema | 0.3748 | **0.4313** | 0.60 |
| Cardiomegaly | 0.3148 | **0.3460** | 0.65 |
| Atelectasis | 0.3090 | **0.3456** | 0.58 |

### Ablation Study

| Variant | Macro AUC | Macro F1 |
|---|---|---|
| True baseline (BCE, no CLAHE, no aug) | 0.7705 | 0.0818 |
| + Focal loss | 0.7707 | 0.0658 |
| + CLAHE | 0.7599 | 0.0754 |
| + Augmentation | **0.8049** | 0.1289 |
| Full pipeline (focal + CLAHE + aug) | 0.7904 | 0.1080 |
| Full pipeline + optimized thresholds | 0.7904 | **0.1927** |

**Key takeaway:** Data augmentation provides the single largest boost to AUC (+3.4 points), while per-class threshold optimization nearly doubles F1 over the default 0.5 threshold.

---

## 🔬 Technical Details

### Image Enhancement Pipeline

```
Grayscale X-ray → Resize(224×224)
    → CLAHE (clip=2.0, 8×8 tiles)
    → Bilateral Denoising (d=5, σ_color=50, σ_space=50)
    → Unsharp Mask Sharpening (amount=1.0)
    → Grayscale-to-RGB replication
    → ImageNet z-score normalization
```

### Data Augmentation (Training Only)

- Horizontal flip (p=0.5)
- Rotation (±7°)
- Random brightness/contrast (±0.1)
- Shift + scale
- Gaussian noise (p=0.3)
- Grid distortion (p=0.2)
- Coarse dropout / cutout (p=0.3)

### Loss Functions

Three loss options for handling severe class imbalance:

| Loss | Description |
|---|---|
| **Asymmetric Loss** (default) | Different γ for positives (γ⁺=1) vs negatives (γ⁻=4) with probability clipping |
| **Focal Loss** | Standard focal loss with per-class weighting |
| **Weighted BCE** | Inverse-frequency weighted binary cross-entropy |

All losses support label smoothing (ε=0.1) and inverse-frequency class weights (clamped at 30×).

### Confidence Estimation

**Monte Carlo Dropout** — runs T=20 stochastic forward passes at inference (dropout layers set to train mode while BatchNorm stays in eval mode). The standard deviation across passes determines confidence:

| Std Range | Confidence Level |
|---|---|
| σ < 0.05 | 🟢 High |
| 0.05 ≤ σ < 0.12 | 🟡 Medium |
| σ ≥ 0.12 | 🔴 Low |

### Anatomical Localization

Grad-CAM++ heatmaps are thresholded at the 80th percentile and overlaid on a **3×2 anatomical grid** (standard radiology convention):

```
┌──────────────┬──────────────┐
│  Right Upper │  Left Upper  │
├──────────────┼──────────────┤
│  Right Mid   │  Left Mid    │
├──────────────┼──────────────┤
│  Right Lower │  Left Lower  │
└──────────────┴──────────────┘
```

Top-k regions (by activation overlap) are reported for each detected disease.

---

## ⚡ Configuration

All hyperparameters are centralized in [`configs/base.yaml`](configs/base.yaml). Key sections:

| Section | Controls |
|---|---|
| `dataset` | Data paths, class names, cache toggle |
| `enhancement` | CLAHE, bilateral filter, sharpening params |
| `augmentation` | Flip/rotation/brightness/noise probabilities |
| `model` | Backbone, Mamba d_model/d_state/num_blocks, fusion hidden, dropout |
| `training` | Loss type, learning rates, scheduler, batch size, epochs, early stopping |
| `confidence` | MC Dropout passes and confidence thresholds |
| `gradcam` | Target layer, alpha blend, heatmap percentile |
| `localization` | Grid dimensions and region names |
| `paths` | Output directories |

---

## 🤝 Acknowledgments

- **NIH Clinical Center** for the ChestX-ray14 dataset
- **Mamba** (Gu & Dao, 2023) for the selective state-space model architecture
- **Asymmetric Loss** (Ben-Baruch et al., 2021) for the multi-label loss function
- **pytorch-grad-cam** for Grad-CAM++ implementation

---

## 📄 License

This project is for academic and research purposes. The NIH ChestX-ray14 dataset is subject to its own [data use agreement](https://nihcc.app.box.com/v/ChestXray-NIHCC).
