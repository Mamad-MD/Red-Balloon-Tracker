import csv
from pathlib import Path

import cv2
import numpy as np

MANIFEST = Path(r"C:\BalloonProject\merged_raw\manifest.csv")
IMG_DIR = Path(r"C:\BalloonProject\dataset_final\images\val")
LBL_DIR = Path(r"C:\BalloonProject\dataset_final\labels\val")
OUT = Path(r"C:\BalloonProject\review\val_tiny_check.jpg")
MAX_SIZE = 20
T = 160

with MANIFEST.open(encoding="utf-8-sig") as f:
    src_of = {row["new_name"]: row["source_folder"] for row in csv.DictReader(f)}

def red_pct(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return -1
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 60)
    return 100 * mask.mean()

tiles = []
for img_path in sorted(IMG_DIR.iterdir()):
    lp = LBL_DIR / (img_path.stem + ".txt")
    if not lp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]
    for line in lp.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        bw, bh = w * W, h * H
        d = max(bw, bh)
        if d > MAX_SIZE:
            continue
        x1, y1 = int((cx - w/2) * W), int((cy - h/2) * H)
        x2, y2 = int((cx + w/2) * W), int((cy + h/2) * H)
        r = red_pct(img, x1, y1, x2, y2)
        src = src_of.get(img_path.stem, "?")
        half = int(max(d, 10) * 2)
        pad = half + 2
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40,40,40))
        px, py = int(cx*W)+pad, int(cy*H)+pad
        crop = cv2.resize(big[py-half:py+half, px-half:px+half], (T, T), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{src} d{d:.0f} r{r:.0f}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        tiles.append(crop)

print(f"{len(tiles)} باکس زیر {MAX_SIZE}px در val")
per_row = 10
pad_n = per_row - (len(tiles) % per_row or per_row)
tiles += [np.zeros((T, T, 3), np.uint8)] * pad_n
rows = [np.hstack(tiles[i:i+per_row]) for i in range(0, len(tiles), per_row)]
cv2.imwrite(str(OUT), np.vstack(rows))
print(f"ذخیره شد: {OUT}")