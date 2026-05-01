"""
ShelfScan inference script.
Input: image path
Output: bounding boxes with class name and confidence
"""

import sys
import json
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    print("Missing deps. Run: pip install ultralytics opencv-python")
    sys.exit(1)

from categories import CLASS_NAMES

DEFAULT_MODEL = Path("models/shelfscan_v1/weights/best.pt")
CONF_THRESHOLD = 0.25


def run_inference(image_path: str, model_path: str = str(DEFAULT_MODEL), save_annotated: bool = True):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model = YOLO(model_path)
    results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
                "confidence": round(conf, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })

    print(json.dumps(detections, indent=2, ensure_ascii=False))

    if save_annotated:
        out_path = img_path.parent / f"{img_path.stem}_detected{img_path.suffix}"
        r = results[0]
        annotated = r.plot()
        cv2.imwrite(str(out_path), annotated)
        print(f"\nAnnotated image saved: {out_path}")

    return detections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)

    model = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_MODEL)
    run_inference(sys.argv[1], model)
