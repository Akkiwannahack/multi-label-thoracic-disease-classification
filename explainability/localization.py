"""
localization.py — Phase 7: Disease Localization

Converts the continuous Grad-CAM++ heatmap into discrete anatomical region
labels using a fixed 3×2 grid approximation of chest anatomy.

Regions:
    Right Upper | Left Upper
    Right Mid   | Left Mid
    Right Lower | Left Lower

Note: "Right" and "Left" are from the patient's perspective (standard
radiology convention) — in a PA chest X-ray, the patient's right is on
the viewer's left.
"""

import cv2
import numpy as np
import pandas as pd


REGION_NAMES_DEFAULT = [
    "Right Upper", "Left Upper",
    "Right Mid",   "Left Mid",
    "Right Lower", "Left Lower",
]


def threshold_heatmap(
    heatmap: np.ndarray,
    percentile: float = 80,
) -> np.ndarray:
    """Threshold the heatmap at the given percentile to get a binary mask.

    Args:
        heatmap: (H, W) float32 in [0, 1]
        percentile: threshold percentile (80th by default)

    Returns:
        Binary mask (H, W) uint8 — 1 where activation exceeds threshold
    """
    threshold = np.percentile(heatmap, percentile)
    mask = (heatmap >= threshold).astype(np.uint8)
    return mask


def make_grid_masks(
    height: int,
    width: int,
    grid_rows: int = 3,
    grid_cols: int = 2,
) -> list[np.ndarray]:
    """Create binary masks for each grid cell.

    Returns list of (H, W) uint8 masks, one per region, in row-major order:
        [Right Upper, Left Upper, Right Mid, Left Mid, Right Lower, Left Lower]
    """
    masks = []
    row_step = height // grid_rows
    col_step = width // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            mask = np.zeros((height, width), dtype=np.uint8)
            r_start = r * row_step
            r_end = (r + 1) * row_step if r < grid_rows - 1 else height
            c_start = c * col_step
            c_end = (c + 1) * col_step if c < grid_cols - 1 else width
            mask[r_start:r_end, c_start:c_end] = 1
            masks.append(mask)

    return masks


def localize(
    heatmap: np.ndarray,
    percentile: float = 80,
    grid_rows: int = 3,
    grid_cols: int = 2,
    region_names: list[str] | None = None,
    top_k: int = 2,
) -> list[str]:
    """Convert a Grad-CAM++ heatmap into anatomical region labels.

    Args:
        heatmap: (H, W) float32 heatmap
        percentile: Threshold percentile for binary mask
        grid_rows: Number of vertical divisions
        grid_cols: Number of horizontal divisions
        region_names: Names for each grid cell (row-major order)
        top_k: Maximum number of regions to return

    Returns:
        List of region name strings (sorted by overlap, descending)
    """
    if region_names is None:
        region_names = REGION_NAMES_DEFAULT

    H, W = heatmap.shape
    binary_mask = threshold_heatmap(heatmap, percentile)
    grid_masks = make_grid_masks(H, W, grid_rows, grid_cols)

    # Compute overlap (intersection area) between binary mask and each grid cell
    overlaps = []
    for i, grid_mask in enumerate(grid_masks):
        intersection = (binary_mask & grid_mask).sum()
        grid_area = grid_mask.sum()
        overlap_ratio = intersection / grid_area if grid_area > 0 else 0.0
        overlaps.append((region_names[i], float(overlap_ratio)))

    # Sort by overlap descending
    overlaps.sort(key=lambda x: x[1], reverse=True)

    # Return regions with non-trivial overlap
    result = []
    for name, ratio in overlaps:
        if ratio > 0 and len(result) < top_k:
            result.append(name)

    # Always return at least one region
    if not result:
        result.append(overlaps[0][0])

    return result


def compute_iou(
    pred_mask: np.ndarray,
    gt_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int] = (224, 224),
) -> float:
    """Compute IoU between a predicted binary mask and a ground-truth bounding box.

    Args:
        pred_mask: (H, W) binary prediction mask
        gt_bbox: (x, y, w, h) ground-truth bounding box (in original image coords)
        image_size: (H, W) of the prediction mask

    Returns:
        IoU score
    """
    H, W = image_size
    gt_mask = np.zeros((H, W), dtype=np.uint8)

    x, y, w, h = gt_bbox
    # Clamp to image bounds
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(W, int(x + w)), min(H, int(y + h))
    gt_mask[y1:y2, x1:x2] = 1

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()

    return float(intersection) / float(union) if union > 0 else 0.0


def run_bbox_sanity_check(
    heatmaps: list[np.ndarray],
    bbox_csv_path: str,
    image_names: list[str],
    original_image_size: tuple[int, int] = (1024, 1024),
    target_size: tuple[int, int] = (224, 224),
    percentile: float = 80,
) -> dict:
    """Phase 7 sanity check: compute IoU against BBox_List_2017.csv ground truth.

    Only ~880 images have bounding boxes (for 8 of 14 classes).

    Returns:
        Summary dict with mean IoU and per-class breakdown
    """
    bbox_df = pd.read_csv(bbox_csv_path)

    results = {}
    ious_all = []

    for idx, (heatmap, img_name) in enumerate(zip(heatmaps, image_names)):
        rows = bbox_df[bbox_df["Image Index"] == img_name]
        if rows.empty:
            continue

        mask = threshold_heatmap(heatmap, percentile)

        for _, row in rows.iterrows():
            # Scale bbox from original image size to target size
            sx = target_size[1] / original_image_size[1]
            sy = target_size[0] / original_image_size[0]

            scaled_bbox = (
                row["Bbox [x"] * sx,
                row["y"] * sy,
                row["w"] * sx,
                row["h]"] * sy,
            )

            iou = compute_iou(mask, scaled_bbox, target_size)
            ious_all.append(iou)

            disease = row["Finding Label"]
            if disease not in results:
                results[disease] = []
            results[disease].append(iou)

    summary = {
        "overall_mean_iou": float(np.mean(ious_all)) if ious_all else 0.0,
        "total_evaluated": len(ious_all),
        "per_class": {},
    }
    for disease, ious in results.items():
        summary["per_class"][disease] = {
            "mean_iou": float(np.mean(ious)),
            "count": len(ious),
        }

    return summary
