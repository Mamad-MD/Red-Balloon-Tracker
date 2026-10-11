import csv, glob, os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\BalloonProject\dataset\Sky")
SKIP_DATASETS = {"sky class.v3i.folder"}
OUT_DIR = Path(r"C:\BalloonProject\real_sky")
OUT_IMG = OUT_DIR / "images"
OUT_IMG.mkdir(parents=True, exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_boxes(lf):
    out = []
    with open(lf, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            out.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return out


rows = []
counter = 0
for dataset_dir in sorted(ROOT.iterdir()):
    if not dataset_dir.is_dir() or dataset_dir.name in SKIP_DATASETS:
        continue
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() not in EXTS:
                continue
            lp = lbl_dir / (p.stem + ".txt")
            if not lp.exists():
                continue
            boxes = read_boxes(lp)
            counter += 1
            new_name = f"realsky_{counter:05d}{p.suffix.lower()}"
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imwrite(str(OUT_IMG / new_name), img)
            box_str = ";".join(f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}" for x1, y1, x2, y2 in boxes)
            rows.append({"file": new_name, "source": dataset_dir.name,
                         "n_objects": len(boxes), "exclude_boxes": box_str})
    print(f"{dataset_dir.name} تمام شد")

with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=["file", "source", "n_objects", "exclude_boxes"])
    wr.writeheader()
    wr.writerows(rows)

clean = sum(1 for r in rows if r["n_objects"] == 0)
with_obj = sum(1 for r in rows if r["n_objects"] > 0)
print(f"\nجمع کل: {len(rows)} عکس | کاملاً خالی: {clean} | دارای هواپیما/پهپاد: {with_obj}")

# کانتکت‌شیت نمونه برای بررسی چشمی
import random
random.seed(0)
sample = random.sample(rows, min(150, len(rows)))
T = 140
tiles = []
for idx, r in enumerate(sample):
    img = cv2.imread(str(OUT_IMG / r["file"]))
    if img is None:
        continue
    tile = cv2.resize(img, (T, T))
    tag = f"{idx}" + ("*" if r["n_objects"] > 0 else "")
    cv2.putText(tile, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    tiles.append(tile)

per_row = 12
pad = per_row - (len(tiles) % per_row or per_row)
tiles += [np.zeros((T, T, 3), np.uint8)] * pad
rows_img = [np.hstack(tiles[j:j+per_row]) for j in range(0, len(tiles), per_row)]
out_review = Path(r"C:\BalloonProject\review\real_sky_sample.jpg")
cv2.imwrite(str(out_review), np.vstack(rows_img))
print(f"کانتکت‌شیت نمونه (۱۵۰ تا) ذخیره شد: {out_review}")
print("علامت * یعنی توی اون عکس هواپیما/پهپاد لیبل شده و ناحیه‌اش موقع چسبوندن بالن نادیده گرفته میشه")