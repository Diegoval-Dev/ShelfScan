"""
End-to-end pipeline: perspective correction → YOLOv8 detection.
Input:  raw shelf image (angled)
Output: warped image + bounding boxes JSON + annotated image

Usage:
    python scripts/detect_on_shelf.py -i data/raw/001.jpg
    python scripts/detect_on_shelf.py -i data/raw/001.jpg -p 50,80 620,90 640,420 30,430
    python scripts/detect_on_shelf.py -d data/raw -o data/results
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from perspective import warp_perspective_image, select_points_interactive, parse_points, auto_detect_corners
from inference import run_inference

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models/shelfscan_v1/weights/best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def process(image_path: Path, points_arg: list[str] | None, output_dir: Path, model: str, auto: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    if points_arg:
        pts = parse_points(points_arg)
    elif auto:
        print(f"Detectando automáticamente esquinas para {image_path.name}...")
        pts = auto_detect_corners(img)
        print(f"Esquinas detectadas: {pts.tolist()}")
    else:
        pts = select_points_interactive(image_path)

    warped = warp_perspective_image(img, pts)
    warped_path = output_dir / f"{image_path.stem}_warped{image_path.suffix}"
    cv2.imwrite(str(warped_path), warped)
    print(f"Warped: {warped_path}")

    detections = run_inference(str(warped_path), model_path=model, save_annotated=True)
    return detections


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("-i", "--image", type=Path)
    target.add_argument("-d", "--dir",   type=Path)
    parser.add_argument("-p", "--points", nargs=4, help="TL TR BR BL as x,y")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/results"))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--auto", action="store_true", help="Detectar automáticamente esquinas del estante")
    args = parser.parse_args()

    if args.image:
        detections = process(args.image, args.points, args.output, args.model, auto=args.auto)
        print(json.dumps(detections, indent=2, ensure_ascii=False))
    else:
        images = [p for p in args.dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        for img in sorted(images):
            print(f"\n--- {img.name} ---")
            try:
                process(img, args.points, args.output, args.model, auto=args.auto)
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    main()
