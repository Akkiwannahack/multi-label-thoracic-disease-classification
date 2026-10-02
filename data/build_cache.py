"""
build_cache.py — Pre-compute enhanced images to an on-disk memmap cache.

Stores a single-channel (grayscale) uint8 memmap of shape (N, H, W) plus a
cache_index.csv that maps image_path → row index.

Usage:
    python -m data.build_cache --config configs/base.yaml [--workers 8]

The cache is resumable: images whose row already exists in cache_index.csv
are skipped on re-run.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path

import cv2
cv2.setNumThreads(0)
import numpy as np
import yaml
from tqdm import tqdm

from data.enhancement import enhance_image


# ── Worker function (top-level for pickling on Windows) ──────────────

def _process_one(
    image_path: str,
    row_idx: int,
    mmap_path: str,
    image_size: int,
    enh_kwargs: dict,
) -> tuple[int, str, bool]:
    """Enhance a single image and write it into the memmap at *row_idx*.

    Returns (row_idx, image_path, success).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (row_idx, image_path, False)

        # enhance_image returns uint8 HWC (3-ch identical grayscale)
        rgb = enhance_image(img, normalize=False, **enh_kwargs)

        # All three channels are identical — keep only one → (H, W) uint8
        gray = rgb[:, :, 0]

        # Open memmap in r+ mode, write this row, flush
        mmap = np.memmap(mmap_path, dtype=np.uint8, mode="r+",
                         shape=(1, image_size, image_size),
                         offset=row_idx * image_size * image_size)
        mmap[0] = gray
        mmap.flush()
        del mmap

        return (row_idx, image_path, True)
    except Exception as e:
        print(f"[cache] ERROR row {row_idx} {image_path}: {e}", file=sys.stderr)
        return (row_idx, image_path, False)


def _process_one_wrapper(args):
    return _process_one(*args)


# ── Main ─────────────────────────────────────────────────────────────

def build_cache(cfg: dict, workers: int = 8) -> None:
    enh = cfg["enhancement"]
    image_size = enh["image_size"]
    enh_kwargs = dict(
        image_size=image_size,
        clahe_clip=enh["clahe_clip_limit"],
        clahe_tile=tuple(enh["clahe_tile_grid"]),
        bilateral_d=enh["bilateral_d"],
        bilateral_sc=enh["bilateral_sigma_color"],
        bilateral_ss=enh["bilateral_sigma_space"],
        sharpen_amount=enh["sharpen_amount"],
    )

    manifest_dir = cfg["paths"]["manifest_dir"]
    cache_dir = os.path.join(cfg["paths"]["output_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ── Collect unique image paths across all splits ──────────────
    import pandas as pd

    all_paths: list[str] = []
    seen = set()
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(manifest_dir, f"manifest_{split}.csv")
        parquet_path = os.path.join(manifest_dir, f"manifest_{split}.parquet")
        if os.path.isfile(parquet_path):
            df = pd.read_parquet(parquet_path)
        elif os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
        else:
            continue
        for p in df["image_path"]:
            if p not in seen:
                seen.add(p)
                all_paths.append(p)

    n_images = len(all_paths)
    print(f"[cache] {n_images} unique images to cache at {image_size}×{image_size}")

    # ── Load or create cache_index.csv ────────────────────────────
    index_path = os.path.join(cache_dir, "cache_index.csv")
    mmap_path = os.path.join(cache_dir, "images.dat")

    existing: dict[str, int] = {}
    if os.path.isfile(index_path):
        with open(index_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["image_path"]] = int(row["row"])
        print(f"[cache] Found existing index with {len(existing)} entries")

    # Build final assignment: keep existing rows, assign new ones
    next_row = max(existing.values(), default=-1) + 1
    assignments: list[tuple[int, str]] = []  # (row_idx, path) for NEW images only
    path_to_row: dict[str, int] = {}

    for p in all_paths:
        if p in existing:
            path_to_row[p] = existing[p]
        else:
            path_to_row[p] = next_row
            assignments.append((next_row, p))
            next_row += 1

    total_rows = next_row
    print(f"[cache] Total rows in memmap: {total_rows}  "
          f"({len(assignments)} new, {len(existing)} already cached)")

    if len(assignments) == 0:
        print("[cache] Nothing to do — cache is already complete.")
        return

    # ── Create / extend memmap file ───────────────────────────────
    expected_bytes = total_rows * image_size * image_size
    if os.path.isfile(mmap_path):
        current_bytes = os.path.getsize(mmap_path)
        if current_bytes < expected_bytes:
            # Extend the file
            with open(mmap_path, "ab") as f:
                f.write(b"\x00" * (expected_bytes - current_bytes))
    else:
        # Create new file
        fp = np.memmap(mmap_path, dtype=np.uint8, mode="w+",
                       shape=(total_rows, image_size, image_size))
        fp.flush()
        del fp

    cache_size_gb = expected_bytes / 1e9
    print(f"[cache] Memmap size: {cache_size_gb:.2f} GB  ({mmap_path})")

    # ── Process images in parallel ────────────────────────────────
    # Prepare (image_path, row_idx, mmap_path, image_size, enh_kwargs) args
    tasks = [(path, row_idx, mmap_path, image_size, enh_kwargs) for row_idx, path in assignments]

    failed = []
    # Write index incrementally so a crash doesn't lose progress
    write_header = not os.path.isfile(index_path)
    index_f = open(index_path, "a", newline="")
    writer = csv.DictWriter(index_f, fieldnames=["image_path", "row"])
    if write_header:
        writer.writeheader()

    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            pbar = tqdm(total=len(tasks), desc="[cache] Building")

            for row_idx, image_path, ok in pool.imap_unordered(_process_one_wrapper, tasks, chunksize=32):
                pbar.update(1)
                if ok:
                    writer.writerow({"image_path": image_path, "row": row_idx})
                else:
                    failed.append(image_path)

                # Flush index every 1000 images for crash safety
                if pbar.n % 1000 == 0:
                    index_f.flush()

            pbar.close()
    finally:
        index_f.flush()
        index_f.close()

    print(f"\n[cache] Done. {len(tasks) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"[cache] Failed images: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build image enhancement cache")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    build_cache(cfg, workers=args.workers)
