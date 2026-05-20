import os
from pathlib import Path

CATEGORIES = {
    0: "bebidas", 1: "lacteos", 2: "snacks", 3: "cereales", 
    4: "limpieza", 5: "enlatados", 6: "aceites", 7: "higiene", 8: "confiteria"
}

AUDIT_DIR = Path("data/planogram_audit_set")
LBL_DIR = Path("data/annotated/labels")
OUT_CSV = Path("data/ground_truth_audit.csv")

lines = ["image_name,category,manual_count,manual_share,manual_breakage"]

for img_path in sorted(AUDIT_DIR.glob("*.jpg")):
    stem = img_path.stem
    lbl_path = LBL_DIR / f"{stem}.txt"
    
    counts = {cat: 0 for cat in CATEGORIES.values()}
    areas = {cat: 0.0 for cat in CATEGORIES.values()}
    
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if not parts: continue
            cls_id = int(parts[0])
            if cls_id in CATEGORIES:
                cat_name = CATEGORIES[cls_id]
                counts[cat_name] += 1
                w, h = float(parts[3]), float(parts[4])
                areas[cat_name] += (w * h)

    for cat_id, cat_name in CATEGORIES.items():
        # Share of shelf (rough estimate from area)
        share = areas[cat_name]
        # Breakage (if count is 0, breakage is 1.0, but let's be more nuanced)
        # For random images, let's just use 0.0 breakage if products exist, 1.0 if not
        breakage = 1.0 if counts[cat_name] == 0 else 0.0
        
        lines.append(f"{img_path.name},{cat_name},{counts[cat_name]},{share:.4f},{breakage:.1f}")

OUT_CSV.write_text("\n".join(lines))
print(f"Generated {OUT_CSV}")
