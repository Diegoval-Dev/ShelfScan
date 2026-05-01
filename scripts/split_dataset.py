"""
Split augmented dataset into train/val/test sets (70/20/10).
Run after augmentation.py.
"""

import os
import shutil
import random
from pathlib import Path

AUGMENTED_DIR = Path("data/augmented")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42

random.seed(SEED)


def split():
    img_dir = AUGMENTED_DIR / "images"
    lbl_dir = AUGMENTED_DIR / "labels"

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        out_img = AUGMENTED_DIR / "images" / split_name
        out_lbl = AUGMENTED_DIR / "labels" / split_name
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy(img_path, out_img / img_path.name)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(lbl_path, out_lbl / lbl_path.name)

        print(f"  {split_name}: {len(split_images)} images")

    print(f"\nTotal: {n} images split into train/val/test")


if __name__ == "__main__":
    split()
