"""
Download labeled images from Open Images v7 (Google) for ShelfScan's weak classes.
No authentication required — fully public dataset.

Maps Open Images categories → ShelfScan classes:
  Bottle, Drink          → bebidas (0)
  Milk, Dairy            → lacteos (1)
  Snack, Cookie, Chips   → snacks (2)
  Cereal, Breakfast food → cereales (3)
  Cleaning agent, Soap   → limpieza (4)
  Tin can                → enlatados (5)
  Cooking oil            → aceites (6)
  Cosmetics, Shampoo     → higiene (7)
  Candy, Chocolate       → confiteria (8)

Usage:
    pip install fiftyone
    python scripts/download_openimages.py
    python scripts/download_openimages.py --samples 100   # per class
"""

import argparse
import shutil
from pathlib import Path

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    raise SystemExit("Run: pip install fiftyone")

ANNOTATED_DIR = Path("data/annotated")

# Open Images class name → our class ID
OI_TO_SHELF: dict[str, int] = {
    "Bottle":           0,
    "Drink":            0,
    "Juice":            0,
    "Milk":             1,
    "Dairy":            1,
    "Snack":            2,
    "Cookie":           2,
    "Potato chip":      2,
    "Breakfast cereal": 3,
    "Cereal":           3,
    "Cleaning agent":   4,
    "Soap":             4,
    "Detergent":        4,
    "Tin can":          5,
    "Canned food":      5,
    "Cooking oil":      6,
    "Vinegar":          6,
    "Cosmetics":        7,
    "Shampoo":          7,
    "Personal care":    7,
    "Candy":            8,
    "Chocolate":        8,
    "Confectionery":    8,
}

# Group by which classes need most data (0 or near-0 in val)
PRIORITY_CLASSES = [
    "Breakfast cereal", "Cereal",
    "Cleaning agent", "Detergent", "Soap",
    "Candy", "Chocolate", "Confectionery",
    "Cooking oil",
    "Cosmetics", "Shampoo",
    "Tin can",
    "Bottle", "Drink",
]


def export_to_yolo(dataset: fo.Dataset, out_img: Path, out_lbl: Path) -> int:
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    exported = 0

    for sample in dataset:
        img_path = Path(sample.filepath)
        if not img_path.exists():
            continue

        detections = sample.get_field("ground_truth")
        if not detections or not detections.detections:
            continue

        lines = []
        for det in detections.detections:
            oi_label = det.label
            cls_id = OI_TO_SHELF.get(oi_label)
            if cls_id is None:
                continue
            x, y, w, h = det.bounding_box  # fiftyone: [top-left-x, top-left-y, w, h] normalized
            cx = x + w / 2
            cy = y + h / 2
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        dst_img = out_img / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        (out_lbl / f"{img_path.stem}.txt").write_text("\n".join(lines))
        exported += 1

    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=75, help="Images per class (default 75)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    out_img = ANNOTATED_DIR / "images"
    out_lbl = ANNOTATED_DIR / "labels"

    total = 0
    for oi_class in PRIORITY_CLASSES:
        print(f"\nDownloading '{oi_class}' ({args.samples} samples)...")
        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=args.split,
                label_types=["detections"],
                classes=[oi_class],
                max_samples=args.samples,
                shuffle=True,
            )
            n = export_to_yolo(ds, out_img, out_lbl)
            print(f"  Exported {n} labeled images → {ANNOTATED_DIR}")
            total += n
            fo.delete_dataset(ds.name)
        except Exception as e:
            print(f"  Failed: {e}")

    # Write classes.txt for our schema
    from categories import CATEGORIES
    (out_lbl / "classes.txt").write_text(
        "\n".join(CATEGORIES[i] for i in sorted(CATEGORIES)) + "\n"
    )

    print(f"\nTotal exported: {total} images with labels")
    print("Next: python scripts/augmentation.py && python scripts/split_dataset.py && python scripts/train.py")


if __name__ == "__main__":
    main()
