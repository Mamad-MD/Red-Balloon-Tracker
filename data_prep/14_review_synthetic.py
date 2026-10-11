import random
from pathlib import Path

import cv2
import numpy as np

IMG_DIR = Path(r"C:\BalloonProject\synthetic_v2\images")
LBL_DIR = Path(r"C:\BalloonProject\synthetic_v2\labels")
OUT = Path(r"C:\BalloonProject\review\synthetic_v2_numbered.jpg")

N = 60
T = 200          # اندازهٔ هر تکه در خروجی
CONTEXT = 60     # چند پیکسل اطراف بالن دیده شود
random.seed(1)

files = sorted(IMG_DIR.glob("*.jpg"))
print(f"{len(files)} عکس مصنوعی پیدا شد")
sample = random.sample(files, min(N, len(files)))

tiles = []
for idx, p in enumerate(sample):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    lp = LBL_DIR / (p.stem + ".txt")
    if img is None or not lp.exists():
        continue
    H, W = img.shape[:2]
    line = lp.read_text().splitlines()[0].split()
    cx, cy, w, h = [float(v) for v in line[1:5]]
    px, py = int(cx * W), int(cy * H)
    bw, bh = int(w * W), int(h * H)

    half = max(bw, bh) // 2 + CONTEXT
    pad = half + 2
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=(40, 40, 40))
    crop = big[py + pad - half: py + pad + half,
               px + pad - half: px + pad + half]
    crop = cv2.resize(crop, (T, T), interpolation=cv2.INTER_NEAREST)

    sc = T / (2 * half)
    cv2.rectangle(crop,
                  (int(T / 2 - bw * sc / 2), int(T / 2 - bh * sc / 2)),
                  (int(T / 2 + bw * sc / 2), int(T / 2 + bh * sc / 2)),
                  (0, 255, 0), 1)
    cv2.putText(crop, f"{idx} d={max(bw, bh)}px", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(crop)

per_row = 10
pad_n = per_row - (len(tiles) % per_row or per_row)
tiles += [np.zeros((T, T, 3), np.uint8)] * pad_n
rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
cv2.imwrite(str(OUT), np.vstack(rows))
print(f"ذخیره شد: {OUT}")