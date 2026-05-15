"""
Plot training curves from YOLOv8 results.csv.

Generates:
  - Loss curves: box, cls, dfl (train + val)
  - Metric curves: mAP@0.5, mAP@0.5:0.95, Precision, Recall

Output: data/results/training_curves/
Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --results models/shelfscan_v1/results.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "models/shelfscan_v1/results.csv"
OUT_DIR = Path("data/results/training_curves")


def plot_losses(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Class Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]

    for ax, (train_col, val_col, title) in zip(axes, loss_pairs):
        ax.plot(df["epoch"], df[train_col], label="Train", color="steelblue")
        ax.plot(df["epoch"], df[val_col], label="Val", color="orange", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Training vs Validation Loss", fontsize=14)
    plt.tight_layout()
    out = OUT_DIR / "loss_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_metrics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = [
        ("metrics/mAP50(B)", "mAP@0.5", "tab:blue"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95", "tab:green"),
        ("metrics/precision(B)", "Precision", "tab:orange"),
        ("metrics/recall(B)", "Recall", "tab:red"),
    ]

    best_epoch = df["metrics/mAP50(B)"].idxmax()
    best_map50 = df.loc[best_epoch, "metrics/mAP50(B)"]

    for ax, (col, label, color) in zip(axes, metrics):
        ax.plot(df["epoch"], df[col], color=color)
        ax.axvline(df.loc[best_epoch, "epoch"], color="gray", linestyle=":", alpha=0.7,
                   label=f"best epoch {int(df.loc[best_epoch, 'epoch'])}")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Training Metrics  |  Best mAP@0.5 = {best_map50:.4f}", fontsize=14)
    plt.tight_layout()
    out = OUT_DIR / "metric_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def print_summary(df: pd.DataFrame) -> None:
    best_idx = df["metrics/mAP50(B)"].idxmax()
    best = df.loc[best_idx]
    print("\n=== Training Summary ===")
    print(f"  Total epochs:       {len(df)}")
    print(f"  Best epoch:         {int(best['epoch'])}")
    print(f"  Best mAP@0.5:       {best['metrics/mAP50(B)']:.4f}")
    print(f"  Best mAP@0.5:0.95:  {best['metrics/mAP50-95(B)']:.4f}")
    print(f"  Precision @ best:   {best['metrics/precision(B)']:.4f}")
    print(f"  Recall @ best:      {best['metrics/recall(B)']:.4f}")
    print(f"  Final train box_loss: {df.iloc[-1]['train/box_loss']:.4f}")
    print(f"  Final val box_loss:   {df.iloc[-1]['val/box_loss']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    csv_path = Path(args.results)
    if not csv_path.exists():
        raise SystemExit(f"results.csv not found: {csv_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, skipinitialspace=True)

    plot_losses(df)
    plot_metrics(df)
    print_summary(df)


if __name__ == "__main__":
    main()
