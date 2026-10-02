import cv2
import torch
import yaml
import matplotlib.pyplot as plt
from models.hybrid_model import build_model
from explainability.gradcam_pp import generate_overlay
from data.enhancement import enhance_image

cfg_path = "configs/base.yaml"
ckpt_path = "checkpoints/best_model.pt"
image_path = r"D:\PBL V\data\raw\images_002\images\00003423_010.png" # Contains Mass (Class 4)
output_path = r"D:\PBL V\thoracic-xray\outputs\mass_heatmap.png"

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(cfg).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load and enhance image
raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
enhanced_rgb = enhance_image(raw_img, normalize=False, image_size=cfg["enhancement"]["image_size"])
enhanced_norm = enhance_image(raw_img, normalize=True, image_size=cfg["enhancement"]["image_size"])

image_tensor = torch.from_numpy(enhanced_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

# Generate Heatmap for Mass (index 4)
heatmap, overlay = generate_overlay(
    model=model,
    image_tensor=image_tensor,
    original_image=enhanced_rgb,
    target_class=4, # Mass
    alpha=0.5,
    device=device
)

# Plot Original vs Heatmap
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(enhanced_rgb)
ax[0].set_title("Enhanced X-Ray")
ax[0].axis('off')

ax[1].imshow(overlay)
ax[1].set_title("Grad-CAM++ (Mass)")
ax[1].axis('off')

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Saved heatmap to {output_path}")
