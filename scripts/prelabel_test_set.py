"""
Pre-label the 45 original shelf images using the current YOLO model.

Generates bootstrap YOLO .txt labels in data/test_relabel/labels/ so
makesense.ai only needs corrections, not from-scratch annotation.

Usage:
    python scripts/prelabel_test_set.py
    python scripts/prelabel_test_set.py --conf 0.20
"""

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Run: pip install ultralytics")

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models/shelfscan_v1/weights/best.pt"
SOURCE_DIR = Path("data/annotated/images")
OUT_DIR = Path("data/test_relabel")

ORIGINAL_PATTERN_LEN = 3


def _is_original(stem: str) -> bool:
    base = stem.split("_")[0]
    return base.isdigit() and len(base) == ORIGINAL_PATTERN_LEN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()

    model = YOLO(args.model)

    images = [p for p in sorted(SOURCE_DIR.glob("*.jpg")) if _is_original(p.stem)]
    if not images:
        raise SystemExit(f"No original images found in {SOURCE_DIR}")

    out_img = OUT_DIR / "images"
    out_lbl = OUT_DIR / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        shutil.copy2(img_path, out_img / img_path.name)

        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        boxes = results[0].boxes
        h, w = results[0].orig_shape

        lines = []
        for box in boxes:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_lbl / f"{img_path.stem}.txt").write_text("\n".join(lines))

    print(f"Pre-labeled {len(images)} images → {OUT_DIR}")
    print("\nNext steps:")
    print("  1. Open https://www.makesense.ai")
    print("  2. Load images from: data/test_relabel/images/")
    print("  3. Actions → Import Annotations → YOLO format → load data/test_relabel/labels/")
    print("  4. Correct wrong/missing boxes (focus on non-enlatados classes)")
    print("  5. Actions → Export Annotations → YOLO format")
    print("  6. Place exported labels into: data/augmented/labels/test/")


if __name__ == "__main__":
    main()
