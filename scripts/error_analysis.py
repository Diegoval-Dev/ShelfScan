"""
Failure case analysis for ShelfScan detector.
Runs inference on test images and groups results by:
  - Confidence distribution per class
  - Low-confidence detections (likely errors)
  - Images with no detections (missed shelf)
  - Per-image detection count

Outputs:
  data/results/error_analysis/  — annotated images + summary plots

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --conf 0.25 --low-conf-threshold 0.35
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Run: pip install ultralytics matplotlib")

sys.path.insert(0, str(Path(__file__).parent))
from categories import CLASS_NAMES

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models/shelfscan_v1/weights/best.pt"
TEST_DIR = Path("data/augmented/images/test")
OUT_DIR = Path("data/results/error_analysis")

COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 230, 14), (60, 143, 255),
    (61, 3, 255), (252, 176, 243),
]


def run_analysis(model_path: str, conf: float, low_conf_threshold: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    images = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))

    if not images:
        raise SystemExit(f"No images in {TEST_DIR}")

    conf_by_class: dict[str, list[float]] = defaultdict(list)
    detections_per_image: list[int] = []
    no_detection_images: list[str] = []
    low_conf_cases: list[dict] = []
    class_counts: dict[str, int] = defaultdict(int)

    for img_path in images:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        r = results[0]
        n = len(r.boxes)
        detections_per_image.append(n)

        if n == 0:
            no_detection_images.append(img_path.name)
            continue

        has_low_conf = False
        for box in r.boxes:
            cls_id = int(box.cls)
            c = float(box.conf)
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            conf_by_class[name].append(c)
            class_counts[name] += 1
            if c < low_conf_threshold:
                has_low_conf = True

        if has_low_conf:
            annotated = r.plot()
            out_path = OUT_DIR / f"lowconf_{img_path.name}"
            cv2.imwrite(str(out_path), annotated)
            low_conf_cases.append({"image": img_path.name, "detections": n})

    _plot_confidence_distributions(conf_by_class)
    _plot_detection_counts(detections_per_image, images)
    _plot_class_counts(class_counts)
    _save_summary(conf_by_class, class_counts, no_detection_images, low_conf_cases, len(images))


def _plot_confidence_distributions(conf_by_class: dict) -> None:
    classes = [c for c in CLASS_NAMES if c in conf_by_class]
    if not classes:
        return

    cols = 3
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = np.array(axes).flatten()

    for i, cls in enumerate(classes):
        confs = conf_by_class[cls]
        axes[i].hist(confs, bins=20, range=(0, 1), color=tuple(c / 255 for c in COLORS[i % len(COLORS)]))
        axes[i].set_title(f"{cls} (n={len(confs)})")
        axes[i].set_xlabel("Confidence")
        axes[i].axvline(np.mean(confs), color="red", linestyle="--", label=f"mean={np.mean(confs):.2f}")
        axes[i].legend(fontsize=7)

    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confidence Distribution per Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_distributions.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/confidence_distributions.png")


def _plot_detection_counts(counts: list[int], images: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(counts)), counts, color="steelblue", alpha=0.7)
    ax.axhline(np.mean(counts), color="red", linestyle="--", label=f"mean={np.mean(counts):.1f}")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Detections")
    ax.set_title("Detections per Image (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "detections_per_image.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/detections_per_image.png")


def _plot_class_counts(class_counts: dict) -> None:
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    colors = [tuple(x / 255 for x in COLORS[CLASS_NAMES.index(c) % len(COLORS)]) if c in CLASS_NAMES else "gray"
              for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, counts, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Total detections")
    ax.set_title("Detection Count per Class (test set)")
    plt.xticks(rotation=30, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/class_distribution.png")


def _save_summary(conf_by_class, class_counts, no_detections, low_conf_cases, total) -> None:
    summary = {
        "total_images": total,
        "images_with_no_detections": no_detections,
        "low_confidence_images": [c["image"] for c in low_conf_cases],
        "mean_confidence_per_class": {
            cls: round(float(np.mean(confs)), 4)
            for cls, confs in conf_by_class.items()
        },
        "detection_count_per_class": dict(class_counts),
        "classes_not_detected": [c for c in CLASS_NAMES if c not in class_counts],
    }

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")

    print("\n=== Error Analysis Summary ===")
    print(f"  Total images:        {total}")
    print(f"  No detections:       {len(no_detections)} — {no_detections}")
    print(f"  Low-conf images:     {len(low_conf_cases)}")
    print(f"  Classes not seen:    {summary['classes_not_detected']}")
    print("\n  Mean confidence per class:")
    for cls, mean_c in summary["mean_confidence_per_class"].items():
        flag = " ⚠" if mean_c < 0.5 else ""
        print(f"    {cls:15s}: {mean_c:.4f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--low-conf-threshold", type=float, default=0.40)
    args = parser.parse_args()
    run_analysis(args.model, args.conf, args.low_conf_threshold)


if __name__ == "__main__":
    main()
