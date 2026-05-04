"""
Download supermarket/grocery datasets from Kaggle.
Copies images to data/raw/ and pre-labeled YOLO annotations to data/annotated/.

Datasets:
  - humansintheloop/supermarket-shelves-dataset   (general shelf images)
  - misahub/supermarket-product-detection         (labeled, various products)
  - tapakah68/grocery-store-dataset               (cereals, oils, cleaning, hygiene)
  - alessandrasala79/supermarket-products         (labeled products)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --only-labeled   # skip unannotated datasets
"""

import argparse
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    raise SystemExit("Run: pip install kagglehub")

RAW_DIR = Path("data/raw")
ANNOTATED_DIR = Path("data/annotated")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Datasets that come with YOLO labels (images + labels/ dirs)
LABELED_DATASETS = [
    "misahub/supermarket-product-detection",
    "tapakah68/grocery-store-dataset",
    "alessandrasala79/supermarket-products",
]

# Datasets with images only (will be auto-labeled later)
UNLABELED_DATASETS = [
    "humansintheloop/supermarket-shelves-dataset",
    "dibyajyotisahoo/supermarket-dataset",
]


def _safe_copy(src: Path, dst: Path) -> bool:
    if dst.exists():
        dst = dst.with_stem(f"{dst.stem}_{src.parent.name}")
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


def copy_images_only(src: Path, dst_img: Path, limit: int | None = None) -> int:
    dst_img.mkdir(parents=True, exist_ok=True)
    images = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if limit:
        images = images[:limit]
    return sum(_safe_copy(img, dst_img / img.name) for img in images)


def copy_labeled(src: Path, dst_img: Path, dst_lbl: Path) -> tuple[int, int]:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    label_dirs = list(src.rglob("labels"))
    if not label_dirs:
        n = copy_images_only(src, dst_img)
        return n, 0

    imgs, lbls = 0, 0
    for lbl_dir in label_dirs:
        img_dir = lbl_dir.parent / "images"
        if not img_dir.exists():
            img_dir = lbl_dir.parent

        for lbl_path in lbl_dir.rglob("*.txt"):
            if lbl_path.name == "classes.txt":
                continue
            img_path = next(
                (img_dir / f"{lbl_path.stem}{ext}" for ext in IMAGE_EXTS
                 if (img_dir / f"{lbl_path.stem}{ext}").exists()),
                None,
            )
            if img_path and _safe_copy(img_path, dst_img / img_path.name):
                shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
                imgs += 1
                lbls += 1

    return imgs, lbls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-labeled", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dst_img = ANNOTATED_DIR / "images"
    dst_lbl = ANNOTATED_DIR / "labels"

    total_imgs, total_lbls = 0, 0

    for dataset_id in LABELED_DATASETS:
        print(f"\nDownloading {dataset_id} ...")
        try:
            path = kagglehub.dataset_download(dataset_id)
            imgs, lbls = copy_labeled(Path(path), dst_img, dst_lbl)
            print(f"  {imgs} images, {lbls} labeled")
            total_imgs += imgs
            total_lbls += lbls
        except Exception as e:
            print(f"  Failed: {e}")

    if not args.only_labeled:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for dataset_id in UNLABELED_DATASETS:
            print(f"\nDownloading {dataset_id} ...")
            try:
                path = kagglehub.dataset_download(dataset_id)
                n = copy_images_only(Path(path), RAW_DIR, args.limit)
                print(f"  {n} images → data/raw/")
                total_imgs += n
            except Exception as e:
                print(f"  Failed: {e}")

    print(f"\nTotal: {total_imgs} images, {total_lbls} with labels")
    print("Next:")
    print("  python scripts/remap_labels.py        # map external classes → our 10")
    print("  python scripts/autolabel.py --input data/raw --output data/annotated  # label raw images")
    print("  python scripts/augmentation.py && python scripts/split_dataset.py && python scripts/train.py")


if __name__ == "__main__":
    main()
