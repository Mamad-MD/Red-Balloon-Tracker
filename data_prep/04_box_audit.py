import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\BalloonProject\dataset\rboflow")
KEEP = [11, 8, 15, 27, 32, 42, 45, 47, 49, 50, 51, 52, 53, 54, 55, 56,
        62, 67, 68, 69, 70, 71]
# پوشه 47: فقط کلاس 4 بالن قرمز است. پوشه 62: کلاس 3. بقیه را بعداً با چشم می‌بینیم
ONLY_CLASS = {47: {4}}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def red_percent(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)
    return 100.0 * mask.mean()


rows = []
for n in KEEP:
    base = ROOT / str(n)
    for split in ("train", "valid", "test"):
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() not in EXTS:
                continue
            lp = lbl_dir / (p.stem + ".txt")
            if not lp.exists():
                continue
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            for parts in lines:
                cls = int(parts[0])
                if n in ONLY_CLASS and cls not in ONLY_CLASS[n]:
                    continue
                vals = list(map(float, parts[1:]))
                if len(vals) == 4:
                    cx, cy, w, h = vals
                    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                else:  # polygon
                    xs, ys = vals[0::2], vals[1::2]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    w, h = x2 - x1, y2 - y1
                bw, bh = w * W, h * H
                outside = x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01
                px1, py1 = max(int(x1 * W), 0), max(int(y1 * H), 0)
                px2, py2 = min(int(x2 * W), W), min(int(y2 * H), H)
                rows.append({
                    "folder": n, "split": split, "file": p.name, "class": cls,
                    "img_w": W, "img_h": H,
                    "box_w_px": round(bw, 1), "box_h_px": round(bh, 1),
                    "area_pct": round(100 * w * h, 3),
                    "aspect": round(bw / max(bh, 1e-6), 2),
                    "red_pct": round(red_percent(img, px1, py1, px2, py2), 1),
                    "outside": int(outside),
                })
    print(f"پوشه {n} تمام شد")

out = Path(r"C:\BalloonProject\box_audit.csv")
with out.open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys())
    wr.writeheader()
    wr.writerows(rows)

# خلاصه برای هر پوشه
print("\nپوشه | کادر | ریز(<4px) | باریک(نسبت>4) | بیرون | قرمز<5% | قرمز>=25%")
for n in KEEP:
    r = [x for x in rows if x["folder"] == n]
    if not r:
        continue
    t = len(r)
    tiny = sum(min(x["box_w_px"], x["box_h_px"]) < 4 for x in r)
    thin = sum(x["aspect"] > 4 or x["aspect"] < 0.25 for x in r)
    out_ = sum(x["outside"] for x in r)
    low = sum(x["red_pct"] < 5 for x in r)
    high = sum(x["red_pct"] >= 25 for x in r)
    pct = lambda k: f"{100*k/t:.0f}%"
    print(f"{n:>4} | {t:>5} | {pct(tiny):>4} | {pct(thin):>4} | {pct(out_):>4} | {pct(low):>4} | {pct(high):>4}")