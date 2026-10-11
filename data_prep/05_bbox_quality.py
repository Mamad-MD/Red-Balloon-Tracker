import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\BalloonProject\dataset\rboflow")
KEEP = [8, 11, 15, 27, 32, 45, 47, 50, 52, 53, 54, 55, 67, 68, 69, 70, 71]
ONLY_CLASS = {47: {4}}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BORDER_MARGIN = 0.01  # ۱٪ حاشیه؛ نزدیک‌تر از این به لبه یعنی "لمس مرز"


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)


def looseness(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0
    mask = red_mask(crop)
    if mask.sum() == 0:
        return -1.0
    ys, xs = np.where(mask)
    rw = xs.max() - xs.min() + 1
    rh = ys.max() - ys.min() + 1
    box_area = (x2 - x1) * (y2 - y1)
    red_box_area = rw * rh
    return red_box_area / max(box_area, 1)


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
                    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                else:
                    xs, ys = vals[0::2], vals[1::2]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                touches_border = (x1 < BORDER_MARGIN or y1 < BORDER_MARGIN or
                                   x2 > 1 - BORDER_MARGIN or y2 > 1 - BORDER_MARGIN)
                px1, py1 = max(int(x1 * W), 0), max(int(y1 * H), 0)
                px2, py2 = min(int(x2 * W), W), min(int(y2 * H), H)
                loose = looseness(img, px1, py1, px2, py2)
                rows.append({
                    "folder": n, "split": split, "file": p.name, "class": cls,
                    "touches_border": int(touches_border),
                    "looseness": round(loose, 2),
                })
    print(f"پوشه {n} تمام شد")

out = Path(r"C:\BalloonProject\bbox_quality.csv")
with out.open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys())
    wr.writeheader()
    wr.writerows(rows)

print("\nپوشه | کادر | لمس‌مرز | خیلی‌شل(<0.3) | رنگ‌ندیده(-1)")
for n in KEEP:
    r = [x for x in rows if x["folder"] == n]
    if not r:
        continue
    t = len(r)
    border = sum(x["touches_border"] for x in r)
    loose = sum(0 <= x["looseness"] < 0.3 for x in r)
    none_ = sum(x["looseness"] == -1 for x in r)
    pct = lambda k: f"{100 * k / t:.0f}%"
    print(f"{n:>4} | {t:>5} | {pct(border):>5} | {pct(loose):>8} | {pct(none_):>6}")