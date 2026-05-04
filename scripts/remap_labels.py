"""
Remap external dataset class IDs to ShelfScan's 10-class schema.

External datasets use different class indices. This script reads their
classes.txt, maps each class name to our schema via keyword matching,
and rewrites all .txt label files with the correct class IDs.

Unmapped classes are dropped (box removed from label file).

Usage:
    python scripts/remap_labels.py
    python scripts/remap_labels.py --dry-run
"""

import argparse
import re
import shutil
from pathlib import Path

from categories import CATEGORIES

LABELS_DIR = Path("data/annotated/labels")

# Keywords per class — order matters, first match wins
CLASS_KEYWORDS: dict[int, list[str]] = {
    0: ["bebida", "beverage", "drink", "soda", "juice", "water", "bottle", "can", "refresco", "agua", "jugo"],
    1: ["lacteo", "dairy", "milk", "leche", "yogurt", "yoghurt", "cheese", "queso", "cream"],
    2: ["snack", "chip", "crisp", "cracker", "cookie", "galleta", "papa", "pretzel"],
    3: ["cereal", "oat", "avena", "granola", "breakfast", "corn flake"],
    4: ["limpieza", "clean", "detergent", "detergente", "soap", "jabon", "disinfect", "bleach", "laundry"],
    5: ["enlatado", "can", "canned", "tin", "tuna", "atun", "bean", "frijol", "sardine"],
    6: ["aceite", "oil", "vinegar", "vinagre", "sauce", "salsa", "condiment"],
    7: ["higiene", "hygiene", "shampoo", "toothpaste", "pasta dental", "deodorant", "desodorante", "soap", "body"],
    8: ["confiteria", "candy", "dulce", "chocolate", "gum", "chicle", "sweet", "confection"],
    9: ["vacio", "empty", "background", "shelf", "estante", "zona"],
}


def match_class(name: str) -> int | None:
    name_lower = name.lower()
    for cls_id, keywords in CLASS_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cls_id
    return None


def build_remap(classes_file: Path) -> dict[int, int] | None:
    if not classes_file.exists():
        return None
    lines = classes_file.read_text().strip().splitlines()
    remap = {}
    for idx, name in enumerate(lines):
        mapped = match_class(name)
        if mapped is not None:
            remap[idx] = mapped
    return remap


def remap_labels_dir(labels_dir: Path, dry_run: bool = False) -> None:
    classes_file = labels_dir / "classes.txt"
    remap = build_remap(classes_file)

    if remap is None:
        print(f"  No classes.txt in {labels_dir} — skipping")
        return

    print(f"  Mapping from {classes_file}:")
    classes = classes_file.read_text().strip().splitlines()
    for ext_id, our_id in remap.items():
        ext_name = classes[ext_id] if ext_id < len(classes) else "?"
        print(f"    {ext_id} ({ext_name}) → {our_id} ({CATEGORIES[our_id]})")

    dropped = [classes[i] for i in range(len(classes)) if i not in remap]
    if dropped:
        print(f"  Dropped (no match): {dropped}")

    txt_files = [f for f in labels_dir.rglob("*.txt") if f.name != "classes.txt"]
    remapped, total_boxes, dropped_boxes = 0, 0, 0

    for txt in txt_files:
        lines = txt.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            ext_id = int(parts[0])
            total_boxes += 1
            if ext_id in remap:
                new_lines.append(f"{remap[ext_id]} " + " ".join(parts[1:]))
            else:
                dropped_boxes += 1

        if not dry_run:
            txt.write_text("\n".join(new_lines))
        remapped += 1

    # Overwrite classes.txt with our schema
    if not dry_run:
        our_classes = "\n".join(CATEGORIES[i] for i in sorted(CATEGORIES)) + "\n"
        classes_file.write_text(our_classes)

    print(f"  {remapped} files processed, {total_boxes - dropped_boxes}/{total_boxes} boxes kept")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files written\n")

    remap_labels_dir(LABELS_DIR, dry_run=args.dry_run)
    print("\nDone. Run augmentation.py next.")


if __name__ == "__main__":
    main()
