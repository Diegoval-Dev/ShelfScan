"""
Formal evaluation of ShelfScan model on test set.
Generates per-class metrics report for Entrega 2.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/shelfscan_v1/weights/best.pt
    python scripts/evaluate.py --iou 0.6 --conf 0.25
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Run: pip install ultralytics")

sys.path.insert(0, str(Path(__file__).parent))
from categories import CATEGORIES, CLASS_NAMES

DATA_YAML = Path("data/dataset.yaml").resolve()
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models/shelfscan_v1/weights/best.pt"


def evaluate(model_path: str, iou: float, conf: float) -> None:
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not DATA_YAML.exists():
        raise SystemExit(f"dataset.yaml not found: {DATA_YAML}")

    model = YOLO(model_path)

    print("=== Evaluating on TEST set ===\n")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        iou=iou,
        conf=conf,
        verbose=True,
    )

    map50      = metrics.box.map50
    map5095    = metrics.box.map
    precision  = metrics.box.mp
    recall     = metrics.box.mr
    per_class  = metrics.box.maps

    print("\n=== ShelfScan v1 — Formal Metrics (Test Set) ===")
    print(f"  mAP@0.5:       {map50:.4f}")
    print(f"  mAP@0.5:0.95:  {map5095:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  IoU threshold: {iou}")
    print(f"  Conf threshold:{conf}")

    print("\n  Per-class mAP@0.5:")
    weak = []
    for i, ap in enumerate(per_class):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        flag = " ⚠" if ap < 0.30 else ""
        print(f"    {name:15s}: {ap:.4f}{flag}")
        if ap < 0.30:
            weak.append(name)

    if weak:
        print(f"\n  Weak classes (<0.30): {', '.join(weak)}")
        print("  → Add more annotated samples for these classes.")

    report_dir = Path(model_path).parent.parent
    report_path = report_dir / "map_report_test.txt"
    with open(report_path, "w") as f:
        f.write("ShelfScan v1 — Formal mAP Report — Test Set (Entrega 2)\n")
        f.write("=" * 55 + "\n")
        f.write(f"mAP@0.5:      {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map5095:.4f}\n")
        f.write(f"Precision:    {precision:.4f}\n")
        f.write(f"Recall:       {recall:.4f}\n")
        f.write(f"IoU:          {iou}\n")
        f.write(f"Conf:         {conf}\n\n")
        f.write("Per-class mAP@0.5:\n")
        for i, ap in enumerate(per_class):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            flag = "  <- WEAK" if ap < 0.30 else ""
            f.write(f"  {name:15s}: {ap:.4f}{flag}\n")
        if weak:
            f.write(f"\nWeak classes: {', '.join(weak)}\n")

    print(f"\nReport saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--iou",  type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    evaluate(args.model, args.iou, args.conf)


if __name__ == "__main__":
    main()
