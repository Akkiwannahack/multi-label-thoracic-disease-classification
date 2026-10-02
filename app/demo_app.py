"""
demo_app.py — Phase 11: Gradio Demo Application

Lightweight web interface for live demonstration:
    Upload X-ray → Enhanced image → Predictions → Grad-CAM++ → Confidence → PDF Report

Usage:
    python -m app.demo_app --config configs/base.yaml \
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import tempfile

import cv2
import numpy as np
import torch
import yaml

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from data.enhancement import enhance_image
from explainability.confidence import predict_with_confidence
from explainability.gradcam_pp import generate_heatmap, overlay_heatmap
from explainability.localization import localize
from models.hybrid_model import build_model
from reporting.generate_report import generate_report


CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load config and model from checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg).to(device)

    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[demo] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print(f"[demo] WARNING: Checkpoint not found at {checkpoint_path} — using random weights!")

    model.eval()
    return model, cfg


def process_xray(
    image: np.ndarray,
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> tuple:
    """Process a single X-ray image through the full pipeline.

    Returns:
        (enhanced_image, predictions_table, heatmap_overlay, report_path)
    """
    enh_cfg = cfg["enhancement"]
    conf_cfg = cfg["confidence"]
    gc_cfg = cfg["gradcam"]
    loc_cfg = cfg["localization"]

    # Convert to grayscale if needed
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image

    # Enhance
    enhanced_rgb = enhance_image(gray, normalize=False,
                                  image_size=enh_cfg["image_size"])
    enhanced_norm = enhance_image(gray, normalize=True,
                                   image_size=enh_cfg["image_size"])

    image_tensor = torch.from_numpy(enhanced_norm).permute(2, 0, 1).float()

    # Predict
    result = predict_with_confidence(
        model, image_tensor,
        class_names=CLASS_NAMES,
        mc_passes=conf_cfg["mc_passes"],
        high_threshold=conf_cfg["high_threshold"],
        medium_threshold=conf_cfg["medium_threshold"],
        device=device,
    )

    # Build predictions table
    table_rows = []
    for d in result["detected_diseases"]:
        cls_name = d["class_name"]

        # Get localization
        heatmap = generate_heatmap(
            model, image_tensor, d["class_index"],
            device=device, method="gradcam++"
        )
        regions = localize(
            heatmap,
            percentile=gc_cfg["heatmap_threshold_percentile"],
            grid_rows=loc_cfg["grid_rows"],
            grid_cols=loc_cfg["grid_cols"],
            region_names=loc_cfg["region_names"],
        )

        table_rows.append([
            cls_name,
            f"{d['probability']:.2%}",
            d["confidence"],
            ", ".join(regions),
        ])

    if not table_rows:
        table_rows = [["No diseases detected", "-", "-", "-"]]

    # Generate heatmap for top prediction
    heatmap_overlay = enhanced_rgb.copy()
    if result["detected_diseases"]:
        top_cls = result["detected_diseases"][0]["class_index"]
        heatmap = generate_heatmap(
            model, image_tensor, top_cls, device=device
        )
        heatmap_overlay = overlay_heatmap(enhanced_rgb, heatmap, alpha=gc_cfg["alpha_blend"])

    # Generate PDF report
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(tmp_img.name, gray_bgr)
        tmp_img_path = tmp_img.name

    report_path = tempfile.mktemp(suffix=".pdf")
    try:
        generate_report(
            tmp_img_path, model, cfg, report_path,
            patient_id="Demo", device=device,
        )
    except Exception as e:
        print(f"[demo] Report generation failed: {e}")
        report_path = None
    finally:
        os.unlink(tmp_img_path)

    return enhanced_rgb, table_rows, heatmap_overlay, report_path


def create_app(model, cfg, device):
    """Create the Gradio interface."""

    def predict(image):
        if image is None:
            return None, [], None, None

        enhanced, table, overlay, report = process_xray(image, model, cfg, device)

        headers = ["Disease", "Probability", "Confidence", "Location"]
        return enhanced, {"headers": headers, "data": table}, overlay, report

    with gr.Blocks(
        title="Thoracic Disease Classifier",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🫁 Thoracic Disease Classification System\n"
            "### ResNet50 + Vision Mamba Hybrid | NIH ChestX-ray14\n"
            "*Upload a chest X-ray to get AI-assisted disease predictions with "
            "Grad-CAM++ heatmaps, confidence scores, and localization.*\n\n"
            "> ⚠️ **Disclaimer**: This is a research prototype for decision support only. "
            "Not a clinical diagnostic tool."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Chest X-ray",
                    type="numpy",
                    height=300,
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tab("📊 Predictions"):
                    predictions_table = gr.Dataframe(
                        headers=["Disease", "Probability", "Confidence", "Location"],
                        label="Detected Diseases",
                    )
                with gr.Tab("🔥 Heatmap"):
                    heatmap_output = gr.Image(label="Grad-CAM++ Heatmap Overlay")
                with gr.Tab("🖼️ Enhanced"):
                    enhanced_output = gr.Image(label="Enhanced X-ray")
                with gr.Tab("📄 Report"):
                    report_output = gr.File(label="Download PDF Report")

        submit_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[enhanced_output, predictions_table, heatmap_output, report_output],
        )

    return app


def main():
    if not GRADIO_AVAILABLE:
        print("Gradio is required for the demo app. Install with: pip install gradio")
        return

    parser = argparse.ArgumentParser(description="Launch demo app")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device: {device}")

    model, cfg = load_model(args.config, args.checkpoint, device)
    app = create_app(model, cfg, device)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
