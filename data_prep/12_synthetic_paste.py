import csv, random
from pathlib import Path

import cv2
import numpy as np

REAL_SKY = Path(r"C:\BalloonProject\real_sky")
OUT_IMG = Path(r"C:\BalloonProject\synthetic\images")
OUT_LBL = Path(r"C:\BalloonProject\synthetic\labels")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

N_PER_SKY = 2
MIN_D, MAX_D = 8, 30
MARGIN = 20  # فاصله‌ی امن (پیکسل) از نواحی exclude
random.seed(0)
np.random.seed(0)


def draw_balloon(diameter):
    size = diameter + 4
    balloon = np.zeros((size, size, 4), np.uint8)
    center = size // 2
    r = diameter // 2
    base_hue = random.randint(0, 8)
    base_sat = random.randint(160, 255)
    base_val = random.randint(120, 220)
    for y in range(size):
        for x in range(size):
            dx, dy = x - center, y - center
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > r:
                continue
            shade = 1 - 0.4 * (dx * -0.4 + dy * -0.4) / max(r, 1)
            shade = np.clip(shade, 0.5, 1.3)
            v = np.clip(base_val * shade, 0, 255)
            hsv = np.uint8([[[base_hue, base_sat, v]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            alpha = 255 if dist < r - 1 else int(255 * (r - dist))
            balloon[y, x] = [*bgr, alpha]
    hl_x = center - int(r * 0.35)
    hl_y = center - int(r * 0.35)
    hl_r = max(int(r * 0.22), 1)
    overlay = balloon.copy()
    cv2.circle(overlay, (hl_x, hl_y), hl_r, (255, 255, 255, 180), -1)
    return cv2.addWeighted(overlay, 0.5, balloon, 0.5, 0)


def paste(sky, balloon, cx, cy):
    bh, bw = balloon.shape[:2]
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = x1 + bw, y1 + bh
    H, W = sky.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    roi = sky[y1:y2, x1:x2].astype(np.float32)
    alpha = balloon[..., 3:4].astype(np.float32) / 255
    fg = balloon[..., :3].astype(np.float32)
    blended = fg * alpha + roi * (1 - alpha)
    sky[y1:y2, x1:x2] = blended.astype(np.uint8)
    return x1, y1, x2, y2


def parse_exclude(s, W, H):
    boxes = []
    if not s:
        return boxes
    for part in s.split(";"):
        if not part:
            continue
        x1, y1, x2, y2 = map(float, part.split(","))
        boxes.append((int(x1*W)-MARGIN, int(y1*H)-MARGIN, int(x2*W)+MARGIN, int(y2*H)+MARGIN))
    return boxes


def overlaps_excluded(cx, cy, d, excl):
    for ex1, ey1, ex2, ey2 in excl:
        if ex1 - d < cx < ex2 + d and ey1 - d < cy < ey2 + d:
            return True
    return False


with (REAL_SKY / "manifest.csv").open(encoding="utf-8-sig") as f:
    sky_rows = list(csv.DictReader(f))

print(f"{len(sky_rows)} عکس آسمان واقعی")
count = 0
for row in sky_rows:
    p = REAL_SKY / "images" / row["file"]
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]
    excl = parse_exclude(row["exclude_boxes"], W, H)
    canvas = img.copy()
    lines = []
    for _ in range(N_PER_SKY):
        d = random.randint(MIN_D, MAX_D)
        for attempt in range(10):
            cx = random.randint(d, W - d)
            cy = random.randint(d, H - d)
            if not overlaps_excluded(cx, cy, d, excl):
                break
        else:
            continue
        balloon = draw_balloon(d)
        box = paste(canvas, balloon, cx, cy)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        ncx, ncy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        nw, nh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
    if not lines:
        continue
    out_name = f"syn_{count:05d}"
    cv2.imwrite(str(OUT_IMG / f"{out_name}.jpg"), canvas)
    (OUT_LBL / f"{out_name}.txt").write_text("\n".join(lines))
    count += 1
    if count % 1000 == 0:
        print(f"  {count} ساخته شد ...")

print(f"\n{count} عکس مصنوعی ساخته شد توی C:\\BalloonProject\\synthetic")