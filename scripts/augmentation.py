"""
Data augmentation pipeline for ShelfScan dataset.
Applies: rotation, brightness shift, blur, horizontal flip.
Input:  data/annotated/  (images + YOLO .txt labels)
Output: data/augmented/  (original + augmented copies)
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")
OUTPUT_DIR = Path("data/augmented")
AUGMENTATIONS_PER_IMAGE = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def load_labels(label_path: Path) -> list[str]:
    if label_path.exists():
        return label_path.read_text().strip().splitlines()
    return []


def save_labels(label_path: Path, lines: list[str]) -> None:
    label_path.write_text("\n".join(lines))


def rotate_image_and_boxes(image: np.ndarray, labels: list[str], angle: float):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        bx, by, bw, bh = map(float, parts[1:])

        # Convert YOLO center to absolute coords
        x1 = (bx - bw / 2) * w
        y1 = (by - bh / 2) * h
        x2 = (bx + bw / 2) * w
        y2 = (by + bh / 2) * h

        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        rotated_corners = (M @ corners_h.T).T

        nx1 = np.clip(rotated_corners[:, 0].min(), 0, w)
        ny1 = np.clip(rotated_corners[:, 1].min(), 0, h)
        nx2 = np.clip(rotated_corners[:, 0].max(), 0, w)
        ny2 = np.clip(rotated_corners[:, 1].max(), 0, h)

        nbx = ((nx1 + nx2) / 2) / w
        nby = ((ny1 + ny2) / 2) / h
        nbw = (nx2 - nx1) / w
        nbh = (ny2 - ny1) / h

        if nbw > 0.01 and nbh > 0.01:
            new_labels.append(f"{cls} {nbx:.6f} {nby:.6f} {nbw:.6f} {nbh:.6f}")

    return rotated, new_labels


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def flip_horizontal(image: np.ndarray, labels: list[str]):
    flipped = cv2.flip(image, 1)
    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        bx, by, bw, bh = map(float, parts[1:])
        bx = 1.0 - bx
        new_labels.append(f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
    return flipped, new_labels


def augment_dataset():
    img_dir = ANNOTATED_DIR / "images"
    lbl_dir = ANNOTATED_DIR / "labels"

    out_img_dir = OUTPUT_DIR / "images"
    out_lbl_dir = OUTPUT_DIR / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    if not image_paths:
        print(f"No images found in {img_dir}. Add images and re-run.")
        return

    total_generated = 0

    for img_path in image_paths:
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.txt"
        labels = load_labels(label_path)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [skip] Cannot read {img_path.name}")
            continue

        # Copy original
        shutil.copy(img_path, out_img_dir / img_path.name)
        save_labels(out_lbl_dir / f"{stem}.txt", labels)

        aug_ops = [
            ("rot_p15",  lambda im, lb: rotate_image_and_boxes(im, lb, 15.0)),
            ("rot_n15",  lambda im, lb: rotate_image_and_boxes(im, lb, -15.0)),
            ("bright_hi", lambda im, lb: (adjust_brightness(im, 1.4), lb)),
            ("bright_lo", lambda im, lb: (adjust_brightness(im, 0.6), lb)),
            ("blur_3",   lambda im, lb: (apply_blur(im, 3), lb)),
            ("blur_5",   lambda im, lb: (apply_blur(im, 5), lb)),
            ("flip",     lambda im, lb: flip_horizontal(im, lb)),
            ("rot_p10_bright", lambda im, lb: (
                adjust_brightness(rotate_image_and_boxes(im, lb, 10.0)[0], 1.2),
                rotate_image_and_boxes(im, lb, 10.0)[1],
            )),
        ]

        selected = random.sample(aug_ops, min(AUGMENTATIONS_PER_IMAGE, len(aug_ops)))

        for name, fn in selected:
            aug_img, aug_labels = fn(image, labels)
            out_name = f"{stem}_{name}"
            cv2.imwrite(str(out_img_dir / f"{out_name}.jpg"), aug_img)
            save_labels(out_lbl_dir / f"{out_name}.txt", aug_labels)
            total_generated += 1

        print(f"  {img_path.name} → {len(selected)} augmented variants")

    total = len(image_paths) + total_generated
    print(f"\nDone. {len(image_paths)} original + {total_generated} augmented = {total} total images")


if __name__ == "__main__":
    augment_dataset()
