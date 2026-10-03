"""
build_cache.py — Pre-compute enhanced images to an on-disk memmap cache.

Stores a single-channel (grayscale) uint8 memmap of shape (N, H, W) plus a
cache_index.csv that maps image_path → row index.

Fix G: cache_meta.json with fingerprint, allocated shape, completed rows.
       Reuse only on fingerprint match.

Usage:
    python -m data.build_cache --config configs/base.yaml [--workers 8]
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
cv2.setNumThreads(0)
import numpy as np
import yaml
from tqdm import tqdm

from data.enhancement import (
    enhance_image,
    get_enhancement_kwargs,
    compute_preprocessing_fingerprint,
)


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
    enh_kwargs = get_enhancement_kwargs(cfg)
    image_size = enh_kwargs["image_size"]

    # Fix G: compute fingerprint
    fingerprint = compute_preprocessing_fingerprint(enh_kwargs)
    print(f"[cache] Preprocessing fingerprint: {fingerprint}")

    manifest_dir = cfg["paths"]["manifest_dir"]
    cache_dir = os.path.join(cfg["paths"]["output_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ── Check for existing cache with different fingerprint ──────
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            old_meta = json.load(f)
        old_fp = old_meta.get("fingerprint", "")
        if old_fp != fingerprint:
            print(
                f"[cache] WARNING: Existing cache has fingerprint '{old_fp}' "
                f"but current config produces '{fingerprint}'. "
                f"The old cache will NOT be reused. "
                f"Use a different cache directory or rebuild."
            )
            # Create a new subdirectory for this fingerprint
            cache_dir = os.path.join(cache_dir, f"fp_{fingerprint}")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"[cache] Using new cache dir: {cache_dir}")

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
        # Still update metadata
        _save_meta(meta_path, fingerprint, total_rows,
                   len(existing), image_size, enh_kwargs)
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
    # Strip imagenet_mean/std from enh_kwargs since we pass normalize=False
    cache_enh_kwargs = {k: v for k, v in enh_kwargs.items()
                        if k not in ("imagenet_mean", "imagenet_std")}
    tasks = [(path, row_idx, mmap_path, image_size, cache_enh_kwargs)
             for row_idx, path in assignments]

    failed = []
    completed = 0
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
                    completed += 1
                else:
                    failed.append(image_path)

                # Flush index every 1000 images for crash safety
                if pbar.n % 1000 == 0:
                    index_f.flush()

            pbar.close()
    finally:
        index_f.flush()
        index_f.close()

    print(f"\n[cache] Done. {completed} succeeded, {len(failed)} failed.")
    if failed:
        print(f"[cache] Failed images: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    # --- Fix G: save metadata ---
    _save_meta(meta_path, fingerprint, total_rows,
               len(existing) + completed, image_size, enh_kwargs)


def _save_meta(meta_path, fingerprint, allocated_rows, completed_rows,
               image_size, enh_kwargs):
    """Save cache_meta.json with fingerprint and allocation info."""
    meta = {
        "fingerprint": fingerprint,
        "allocated_rows": allocated_rows,
        "completed_rows": completed_rows,
        "image_size": image_size,
        "dtype": "uint8",
        "enhancement_params": {
            k: v for k, v in enh_kwargs.items()
            if k not in ("imagenet_mean", "imagenet_std")
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] Saved cache metadata → {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build image enhancement cache")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    build_cache(cfg, workers=args.workers)
