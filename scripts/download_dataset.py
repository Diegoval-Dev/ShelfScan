"""
Download supermarket shelf images from Kaggle and copy them to data/raw/.

Datasets downloaded (in order of priority):
  1. humansintheloop/supermarket-shelves-dataset  (~600 shelf images)
  2. dibyajyotisahoo/supermarket-dataset          (fallback, extra variety)

Requirements:
    pip install kagglehub
    Kaggle credentials in ~/.kaggle/kaggle.json  OR  env vars:
        KAGGLE_USERNAME and KAGGLE_KEY

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --limit 200   # copy only first N images
"""

import argparse
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    raise SystemExit("kagglehub not installed. Run: pip install kagglehub")

RAW_DIR = Path("data/raw")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DATASETS = [
    "humansintheloop/supermarket-shelves-dataset",
    "dibyajyotisahoo/supermarket-dataset",
]


def copy_images(src: Path, dst: Path, limit: int | None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    images = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if limit:
        images = images[:limit]

    copied = 0
    for img in images:
        target = dst / img.name
        if target.exists():
            # avoid name collisions
            target = dst / f"{img.stem}_{img.parent.name}{img.suffix}"
        shutil.copy2(img, target)
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max images to copy into data/raw/")
    args = parser.parse_args()

    total = 0
    for dataset_id in DATASETS:
        print(f"\nDownloading {dataset_id} ...")
        try:
            path = kagglehub.dataset_download(dataset_id)
            n = copy_images(Path(path), RAW_DIR, args.limit)
            print(f"  Copied {n} images → data/raw/")
            total += n
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"\nTotal images in data/raw/: {total}")
    print("Next step: python scripts/autolabel.py --input data/raw --output data/annotated")


if __name__ == "__main__":
    main()
