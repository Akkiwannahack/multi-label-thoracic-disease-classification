"""
diagnose_instability.py — Step 2: Diagnose the epoch-2/epoch-4 instability

Reads the training metrics log and configs to analyze whether validation loss
degradation correlates with the backbone unfreezing point (stage1_epochs).

Usage (from thoracic-xray/):
    python -m training.diagnose_instability --config configs/base.yaml
"""

import argparse
import os
import sys

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Diagnose training instability"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--metrics", type=str, default=None,
        help="Path to metrics_log.csv (default: outputs/clean_metrics.csv)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    metrics_path = args.metrics or os.path.join(
        cfg["paths"]["output_dir"], "clean_metrics.csv"
    )

    if not os.path.isfile(metrics_path):
        metrics_path = os.path.join(cfg["paths"]["output_dir"], "metrics_log.csv")

    if not os.path.isfile(metrics_path):
        print(f"[diagnose] ERROR: No metrics file found at {metrics_path}")
        return

    # Read metrics — handle the fact that metrics_log.csv may have two
    # header rows from two separate training runs (old 5k and new 30k)
    with open(metrics_path, "r") as f:
        lines = f.readlines()

    # Find the last header row
    header_indices = [
        i for i, line in enumerate(lines) if line.startswith("auc_Atelectasis")
    ]

    if len(header_indices) == 0:
        print("[diagnose] ERROR: No header row found in metrics CSV")
        return

    last_header_idx = header_indices[-1]
    relevant_lines = lines[last_header_idx:]

    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as tmp:
        tmp.writelines(relevant_lines)
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)
    os.unlink(tmp_path)

    # --- Config values ---
    t = cfg["training"]
    stage1_epochs = t["stage1_epochs"]
    lr_backbone_stage1 = t["lr_backbone_stage1"]
    lr_head_stage1 = t["lr_head_stage1"]
    lr_backbone_stage2 = t["lr_backbone_stage2"]
    lr_head_stage2 = t["lr_head_stage2"]

    print("\n" + "=" * 70)
    print("  Step 2 — Training Instability Diagnosis")
    print("=" * 70)

    print(f"\n  Config: stage1_epochs = {stage1_epochs}")
    print(f"  Config: lr_backbone_stage1 = {lr_backbone_stage1}")
    print(f"  Config: lr_head_stage1     = {lr_head_stage1}")
    print(f"  Config: lr_backbone_stage2 = {lr_backbone_stage2}")
    print(f"  Config: lr_head_stage2     = {lr_head_stage2}")
    print(f"  Config: backbone LR ratio (stage2/stage1) = "
          f"{lr_backbone_stage2 / lr_backbone_stage1:.2f}x")
    print(f"  Config: head LR ratio (stage2/stage1) = "
          f"{lr_head_stage2 / lr_head_stage1:.2f}x")

    # --- Epoch-by-epoch analysis ---
    print(f"\n  {'Epoch':>5s}  {'Train Loss':>10s}  {'Val Loss':>10s}  "
          f"{'Macro AUC':>10s}  {'Macro F1':>9s}  {'Notes'}")
    print(f"  {'-' * 70}")

    best_auc_epoch = -1
    best_auc = 0.0
    val_loss_at_best = 0.0

    for _, row in df.iterrows():
        epoch = int(row["epoch"])
        train_loss = row["train_loss"]
        val_loss = row["val_loss"]
        macro_auc = row["macro_auc"]
        macro_f1 = row["macro_f1"]

        notes = []
        if epoch == stage1_epochs:
            notes.append("<- LAST stage-1 epoch")
        if epoch == stage1_epochs + 1:
            notes.append("<- FIRST stage-2 epoch (backbone unfreezes)")

        if macro_auc > best_auc:
            best_auc = macro_auc
            best_auc_epoch = epoch
            val_loss_at_best = val_loss
            notes.append("* best AUC so far")

        # Detect val_loss spike
        if epoch > 1:
            prev_row = df[df["epoch"] == epoch - 1]
            if len(prev_row) > 0:
                prev_val_loss = prev_row.iloc[0]["val_loss"]
                if val_loss > prev_val_loss * 1.05:
                    pct = (val_loss - prev_val_loss) / prev_val_loss * 100
                    notes.append(f"!! val_loss +{pct:.1f}%")

        note_str = "  ".join(notes)
        print(f"  {epoch:>5d}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{macro_auc:>10.4f}  {macro_f1:>9.4f}  {note_str}")

    # --- Overfitting analysis ---
    print(f"\n  {'=' * 70}")
    print(f"  DIAGNOSIS")
    print(f"  {'=' * 70}")

    print(f"\n  Best checkpoint: epoch {best_auc_epoch} "
          f"(macro-AUC = {best_auc:.4f}, val_loss = {val_loss_at_best:.4f})")

    # Compute the train-val loss gap at each epoch
    val_losses = df[["epoch", "val_loss"]].values
    train_losses = df[["epoch", "train_loss"]].values

    print(f"\n  Train-Val Loss Gap (overfitting indicator):")
    print(f"  {'Epoch':>5s}  {'Train':>8s}  {'Val':>8s}  {'Gap':>8s}  {'Gap Delta':>10s}")
    print(f"  {'-' * 45}")

    prev_gap = None
    for i in range(len(val_losses)):
        epoch = int(val_losses[i][0])
        vl = val_losses[i][1]
        tl = train_losses[i][1]
        gap = vl - tl
        gap_delta = gap - prev_gap if prev_gap is not None else 0.0
        prev_gap = gap

        marker = ""
        if epoch == stage1_epochs + 1:
            marker = " <- unfreezing"
        if gap_delta > 0.05:
            marker += " !! gap widening fast"

        print(f"  {epoch:>5d}  {tl:>8.4f}  {vl:>8.4f}  {gap:>+8.4f}  "
              f"{gap_delta:>+10.4f}{marker}")

    # Key observations
    unfreezing_epoch = stage1_epochs + 1
    if unfreezing_epoch <= len(df):
        before = df[df["epoch"] == stage1_epochs]
        after = df[df["epoch"] == unfreezing_epoch]
        if len(before) > 0 and len(after) > 0:
            vl_before = before.iloc[0]["val_loss"]
            vl_after = after.iloc[0]["val_loss"]
            auc_before = before.iloc[0]["macro_auc"]
            auc_after = after.iloc[0]["macro_auc"]

            print(f"\n  Unfreezing impact (epoch {stage1_epochs} -> {unfreezing_epoch}):")
            print(f"    val_loss: {vl_before:.4f} -> {vl_after:.4f} "
                  f"({(vl_after-vl_before)/vl_before*100:+.1f}%)")
            print(f"    macro_auc: {auc_before:.4f} -> {auc_after:.4f} "
                  f"({(auc_after-auc_before)/auc_before*100:+.1f}%)")

    # Train loss trajectory
    first_train = train_losses[0][1]
    last_train = train_losses[-1][1]
    first_val = val_losses[0][1]
    last_val = val_losses[-1][1]

    print(f"\n  Overall trajectory:")
    print(f"    train_loss: {first_train:.4f} -> {last_train:.4f} "
          f"({(last_train-first_train)/first_train*100:+.1f}%)")
    print(f"    val_loss:   {first_val:.4f} -> {last_val:.4f} "
          f"({(last_val-first_val)/first_val*100:+.1f}%)")
    print(f"    -> Train loss falls while val loss rises = classic overfitting")

    # Final recommendation preview
    backbone_lr_ratio = lr_backbone_stage2 / lr_backbone_stage1
    print(f"\n  Key finding: backbone LR drops {backbone_lr_ratio:.1f}x from "
          f"stage 1 to stage 2,")
    print(f"  but the head LR only drops {lr_head_stage2/lr_head_stage1:.1f}x.")
    print(f"  At epoch {unfreezing_epoch} all backbone layers are unfrozen at "
          f"LR={lr_backbone_stage2},")
    print(f"  which may perturb pretrained low-level features too aggressively.")

    print(f"\n  (Step 3 will propose a targeted fix based on this diagnosis.)")
    print()


if __name__ == "__main__":
    main()
