"""
Split augmented dataset into train/val/test sets.

Strategy:
  - Original shelf images (001.jpg..045.jpg) → test set (clean holdout, never mixed)
  - Open Images / other sources → train + val only (80/20)

This keeps test set distribution stable across re-trainings.
"""

import shutil
import random
from pathlib import Path

AUGMENTED_DIR = Path("data/augmented")
SEED = 42
random.seed(SEED)


def _is_original(stem: str) -> bool:
    """Original humansintheloop images are named 001–045 (with optional augmentation suffix)."""
    base = stem.split("_")[0]
    return base.isdigit() and len(base) == 3


def split():
    img_dir = AUGMENTED_DIR / "images"
    lbl_dir = AUGMENTED_DIR / "labels"

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    original = [p for p in all_images if _is_original(p.stem)]
    external = [p for p in all_images if not _is_original(p.stem)]

    random.shuffle(external)
    n_ext = len(external)
    n_val = int(n_ext * 0.20)

    splits = {
        "train": external[n_val:],
        "val":   external[:n_val],
        "test":  original,
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

    print(f"\nOriginal shelf images in test ({len(original)}), "
          f"external data in train/val ({len(external)})")


if __name__ == "__main__":
    split()
