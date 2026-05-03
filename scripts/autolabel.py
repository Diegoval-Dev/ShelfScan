"""
Zero-shot pre-labeling using YOLO-World.
Generates YOLO-format .txt label files for images in a directory.
Output is meant to be reviewed and corrected with labelImg before training.

Usage:
    python scripts/autolabel.py --input data/raw --output data/annotated
"""

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLOWorld
except ImportError:
    raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

from categories import CLASS_NAMES

TEXT_PROMPTS = [
    "beverage bottle or can on shelf",     # bebidas
    "dairy product milk yogurt cheese",    # lacteos
    "snack bag chips crackers cookies",    # snacks
    "cereal box breakfast",                # cereales
    "cleaning product detergent bottle",   # limpieza
    "canned food tin can",                 # enlatados
    "cooking oil bottle sauce vinegar",    # aceites
    "hygiene product shampoo toothpaste",  # higiene
    "candy chocolate confectionery",       # confiteria
    "empty shelf space no product",        # zona_vacia
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def autolabel_directory(
    input_dir: str,
    output_dir: str,
    conf: float = 0.15,
    model_id: str = "yolov8s-worldv2.pt",
) -> None:
    input_path = Path(input_dir)
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"No images found in {input_dir}")
        return

    model = YOLOWorld(model_id)
    model.set_classes(TEXT_PROMPTS)

    # labelImg requires classes.txt inside the labels directory
    (out_lbl_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    total_boxes = 0
    for img_path in sorted(images):
        results = model.predict(str(img_path), conf=conf, verbose=False)
        r = results[0]
        h, w = r.orig_shape

        lines = []
        for box in r.boxes:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bx = ((x1 + x2) / 2) / w
            by = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

        lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
        lbl_path.write_text("\n".join(lines))
        shutil.copy(img_path, out_img_dir / img_path.name)

        total_boxes += len(lines)
        print(f"  {img_path.name}: {len(lines)} boxes")

    print(f"\nDone. {len(images)} images, {total_boxes} total boxes.")
    print(f"Labels in: {out_lbl_dir}")
    print("Next: open labelImg to review and correct labels before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot pre-labeling with YOLO-World.")
    parser.add_argument("--input", required=True, help="Directory with raw images.")
    parser.add_argument("--output", required=True, help="Output directory for annotated images and labels.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15).")
    parser.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO-World model variant.")
    args = parser.parse_args()

    autolabel_directory(args.input, args.output, args.conf, args.model)


if __name__ == "__main__":
    main()
