"""
generate_report.py  -  Phase 9: Diagnostic Report Generation

Assembles predictions, Grad-CAM++ heatmaps, confidence estimates, and
localization into a doctor-friendly PDF report.

Fix H: Uses fitted thresholds (not hard-coded 0.5). Reports which
       thresholds are in use. Full enhancement config from checkpoint.

Usage:
    python -m reporting.generate_report --config configs/base.yaml \\
        --checkpoint checkpoints/best_model.pt \\
        --image path/to/xray.png \\
        --output outputs/reports/report.pdf
"""

import argparse
import json
import os
import tempfile

import cv2
import numpy as np
import torch
import yaml
from fpdf import FPDF

from data.enhancement import enhance_image, get_enhancement_kwargs
from explainability.confidence import predict_with_confidence
from explainability.gradcam_pp import generate_heatmap, overlay_heatmap
from explainability.localization import localize
from models.hybrid_model import build_model


CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

RECOMMENDATIONS = {
    "Atelectasis": "Findings suggest Atelectasis in the {region}  -  recommend radiologist confirmation.",
    "Cardiomegaly": "Findings suggest Cardiomegaly in the {region}  -  recommend echocardiography.",
    "Effusion": "Findings suggest Pleural Effusion in the {region}  -  recommend lateral decubitus X-ray.",
    "Infiltration": "Findings suggest Infiltration in the {region}  -  recommend clinical correlation.",
    "Mass": "Findings suggest a Mass in the {region}  -  recommend CT follow-up.",
    "Nodule": "Findings suggest a Nodule in the {region}  -  recommend CT follow-up per Fleischner criteria.",
    "Pneumonia": "Findings suggest Pneumonia in the {region}  -  recommend clinical correlation and labs.",
    "Pneumothorax": "Findings suggest Pneumothorax in the {region}  -  urgent clinical review recommended.",
    "Consolidation": "Findings suggest Consolidation in the {region}  -  recommend clinical correlation.",
    "Edema": "Findings suggest Pulmonary Edema in the {region}  -  recommend echocardiography.",
    "Emphysema": "Findings suggest Emphysema in the {region}  -  recommend PFTs.",
    "Fibrosis": "Findings suggest Fibrosis in the {region}  -  recommend HRCT.",
    "Pleural_Thickening": "Findings suggest Pleural Thickening in the {region}  -  recommend CT follow-up.",
    "Hernia": "Findings suggest Hernia in the {region}  -  recommend surgical consultation.",
}


class DiagnosticReport(FPDF):
    """PDF report for thoracic disease classification results."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Thoracic Disease Classification  -  Diagnostic Report", ln=True, align="C")
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, "AI-assisted analysis  -  not a clinical diagnosis. "
                  "For decision support only.", ln=True, align="C")
        self.ln(5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(
    image_path: str,
    model: torch.nn.Module,
    cfg: dict,
    output_path: str,
    patient_id: str = "Unknown",
    device: torch.device | None = None,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Generate a full diagnostic PDF report for a single X-ray image.

    Fix H: Uses fitted thresholds. Full enhancement config from cfg.

    Args:
        image_path: Path to the input chest X-ray image
        model: Trained HybridResNetMamba model
        cfg: Configuration dictionary
        output_path: Where to save the PDF
        patient_id: Patient identifier
        device: Torch device
        thresholds: Per-class fitted thresholds (Fix H). If None, uses 0.5.

    Returns:
        Path to the generated PDF
    """
    if device is None:
        device = next(model.parameters()).device

    enh_kwargs = get_enhancement_kwargs(cfg)
    conf_cfg = cfg["confidence"]
    gc_cfg = cfg["gradcam"]
    loc_cfg = cfg["localization"]

    # --- Load and enhance image ---
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Enhanced (uint8 RGB for visualization)
    enhanced_rgb = enhance_image(raw_img, normalize=False, **enh_kwargs)

    # Enhanced (float32 normalized for model input)
    enhanced_norm = enhance_image(raw_img, normalize=True, **enh_kwargs)
    image_tensor = torch.from_numpy(enhanced_norm).permute(2, 0, 1).float()

    # --- Predict with confidence (Fix H: uses fitted thresholds) ---
    result = predict_with_confidence(
        model, image_tensor,
        class_names=CLASS_NAMES,
        mc_passes=conf_cfg["mc_passes"],
        high_threshold=conf_cfg["high_threshold"],
        medium_threshold=conf_cfg["medium_threshold"],
        device=device,
        thresholds=thresholds,
    )

    detected = result["detected_diseases"]

    # --- Generate Grad-CAM++ and localization for detected diseases ---
    heatmap_overlays = {}
    localizations = {}

    for disease_info in detected:
        cls_idx = disease_info["class_index"]
        cls_name = disease_info["class_name"]

        heatmap = generate_heatmap(
            model, image_tensor, cls_idx, device=device, method="gradcam++"
        )

        overlay = overlay_heatmap(
            enhanced_rgb, heatmap, alpha=gc_cfg["alpha_blend"]
        )
        heatmap_overlays[cls_name] = overlay

        regions = localize(
            heatmap,
            percentile=gc_cfg["heatmap_threshold_percentile"],
            grid_rows=loc_cfg["grid_rows"],
            grid_cols=loc_cfg["grid_cols"],
            region_names=loc_cfg["region_names"],
        )
        localizations[cls_name] = regions

    # --- Build PDF ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pdf = DiagnosticReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Patient info
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
    image_name = os.path.basename(image_path)
    pdf.cell(0, 8, f"Image: {image_name}", ln=True)

    # Fix H: report which thresholds are in use
    if thresholds:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 6, "Using validation-fitted per-class thresholds.", ln=True)
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 6, "Using default 0.5 threshold for all classes.", ln=True)
    pdf.ln(5)

    # Original vs Enhanced image
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Enhanced X-ray Image:", ln=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp.name, enhanced_bgr)
    pdf.image(tmp.name, x=10, w=80)
    os.unlink(tmp.name)

    pdf.ln(5)

    # No findings
    if not detected:
        pdf.set_font("Helvetica", "", 11)
        thr_note = "validation-fitted" if thresholds else "0.5"
        pdf.cell(0, 8, f"No diseases detected (all probabilities below {thr_note} threshold).", ln=True)
        pdf.output(output_path)
        return output_path

    # Findings table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Detected Diseases:", ln=True)
    pdf.ln(2)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [45, 25, 25, 40, 55]
    headers = ["Disease", "Probability", "Confidence", "Location", "Recommendation"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    for disease_info in detected:
        cls_name = disease_info["class_name"]
        prob = f"{disease_info['probability']:.2%}"
        conf = disease_info["confidence"]
        region_str = ", ".join(localizations.get(cls_name, ["N/A"]))

        rec_template = RECOMMENDATIONS.get(cls_name, "Recommend clinical review.")
        recommendation = rec_template.format(region=region_str)

        row_data = [cls_name, prob, conf, region_str, recommendation]
        for i, val in enumerate(row_data):
            pdf.cell(col_widths[i], 8, val, border=1)
        pdf.ln()

    pdf.ln(5)

    # Heatmap overlays
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Grad-CAM++ Heatmap Overlays:", ln=True)
    pdf.ln(2)

    for cls_name, overlay in heatmap_overlays.items():
        pdf.set_font("Helvetica", "B", 9)
        region_str = ", ".join(localizations.get(cls_name, []))
        pdf.cell(0, 7, f"{cls_name}  -  Region: {region_str}", ln=True)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp.name, overlay_bgr)
        pdf.image(tmp.name, x=10, w=80)
        os.unlink(tmp.name)

        pdf.ln(3)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4,
                   "DISCLAIMER: This report is generated by an AI system for decision "
                   "support purposes only. It is NOT a clinical diagnosis. All findings "
                   "should be reviewed and confirmed by a qualified radiologist or "
                   "physician before any clinical action is taken.")

    pdf.output(output_path)
    print(f"[report] Saved diagnostic report -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic report")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--output", type=str, default="outputs/reports/report.pdf")
    parser.add_argument("--patient-id", type=str, default="Unknown")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Path to optimized_thresholds.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg).to(device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load thresholds if provided
    thresholds = None
    if args.thresholds:
        with open(args.thresholds, "r") as f:
            thr_data = json.load(f)
        thresholds = {}
        for name in CLASS_NAMES:
            val = thr_data.get(name, {"threshold": 0.5})
            thresholds[name] = val["threshold"] if isinstance(val, dict) else val

    generate_report(
        args.image, model, cfg, args.output,
        patient_id=args.patient_id, device=device,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
