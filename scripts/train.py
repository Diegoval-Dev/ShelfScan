"""
YOLOv8 training script for ShelfScan.
Preliminary run for Entrega 1 — uses yolov8n (nano) for fast iteration.
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

DATA_YAML = Path("data/dataset.yaml").resolve()
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def train():
    if not DATA_YAML.exists():
        print(f"dataset.yaml not found at {DATA_YAML}")
        sys.exit(1)

    device = _best_device()

    train_config = {
        "data": str(DATA_YAML),
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "patience": 10,
        "project": str(MODEL_DIR),
        "name": "shelfscan_v1",
        "exist_ok": True,
        "plots": True,
        "save": True,
        "device": device,
        "amp": device != "mps",
    }

    print(f"Device: {device}")
    print(f"Config: {train_config}")

    model = YOLO("yolov8n.pt")
    results = model.train(**train_config)

    print("\n=== Training complete ===")
    print(f"Best model: {MODEL_DIR}/shelfscan_v1/weights/best.pt")

    # Quick validation pass
    metrics = model.val()
    print(f"\nmAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    report_path = results.save_dir / "map_report.txt"
    with open(report_path, "w") as f:
        f.write("ShelfScan v1 — Preliminary mAP Report (Entrega 1)\n")
        f.write("=" * 50 + "\n")
        f.write(f"mAP@0.5:      {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Precision:    {metrics.box.mp:.4f}\n")
        f.write(f"Recall:       {metrics.box.mr:.4f}\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    train()
