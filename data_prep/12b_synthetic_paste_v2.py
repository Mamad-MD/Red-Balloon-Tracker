import csv, random
from pathlib import Path

import cv2
import numpy as np

REAL_SKY = Path(r"C:\BalloonProject\real_sky")
OUT_IMG = Path(r"C:\BalloonProject\synthetic_v2\images")
OUT_LBL = Path(r"C:\BalloonProject\synthetic_v2\labels")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

N_PER_SKY = 2
MIN_D, MAX_D = 8, 30
MARGIN = 20
random.seed(0)
np.random.seed(0)


def make_balloon(d):
    """بالن را ۴ برابر بزرگ‌تر می‌کشد و کوچک می‌کند تا لبه نرم شود."""
    S = 4
    size = (d + 4) * S
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    rx = d * S / 2
    ry = rx * random.uniform(0.9, 1.1)
    nx, ny = (xx - c) / rx, (yy - c) / ry
    dist = np.sqrt(nx ** 2 + ny ** 2)
    mask = (dist <= 1).astype(np.float32)

    hue = random.choice([random.randint(0, 4), random.randint(175, 179)])
    sat = random.randint(160, 255)
    base_val = random.randint(140, 225)
    shade = np.clip(1.0 - 0.25 * (nx + ny), 0.6, 1.3) * (1 - 0.25 * dist ** 2)
    val = np.clip(base_val * shade, 0, 255)
    hsv = np.stack([np.full_like(val, hue), np.full_like(val, sat), val], -1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    # هایلایت نرم، فقط گاهی و فقط برای بالن‌های بزرگ‌تر
    if d >= 18 and random.random() < 0.5:
        hx, hy = c - 0.35 * rx, c - 0.35 * ry
        sig = 0.18 * rx
        hl = np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sig ** 2)))[..., None]
        bgr = bgr * (1 - 0.4 * hl) + 255 * 0.4 * hl

    n = d + 4
    bgr = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(mask, (n, n), interpolation=cv2.INTER_AREA)
    sigma = random.uniform(0.4, 1.0)
    bgr = cv2.GaussianBlur(bgr, (0, 0), sigma)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma * 0.6)
    return bgr, alpha


def sky_ok(img, cx, cy, d):
    r = 3 * d
    H, W = img.shape[:2]
    if cx - r < 0 or cy - r < 0 or cx + r > W or cy + r > H:
        return False, 0
    patch = img[cy - r:cy + r, cx - r:cx + r]
    v = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 2].mean()
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return (v >= 80 and gray.std() <= 30), float(gray.std())


def paste(canvas, bgr, alpha, cx, cy, noise_std):
    n = bgr.shape[0]
    x0, y0 = cx - n // 2, cy - n // 2
    roi = canvas[y0:y0 + n, x0:x0 + n].astype(np.float32)
    fg = bgr + np.random.normal(0, noise_std, bgr.shape).astype(np.float32)
    a = alpha[..., None]
    canvas[y0:y0 + n, x0:x0 + n] = np.clip(fg * a + roi * (1 - a), 0, 255).astype(np.uint8)
    ys, xs = np.where(alpha > 0.3)
    return x0 + xs.min() - 1, y0 + ys.min() - 1, x0 + xs.max() + 2, y0 + ys.max() + 2


def parse_exclude(s, W, H):
    boxes = []
    for part in (s or "").split(";"):
        if part:
            x1, y1, x2, y2 = map(float, part.split(","))
            boxes.append((int(x1 * W) - MARGIN, int(y1 * H) - MARGIN,
                          int(x2 * W) + MARGIN, int(y2 * H) + MARGIN))
    return boxes


def in_excluded(cx, cy, d, excl):
    return any(ex1 - d < cx < ex2 + d and ey1 - d < cy < ey2 + d for ex1, ey1, ex2, ey2 in excl)


with (REAL_SKY / "manifest.csv").open(encoding="utf-8-sig") as f:
    sky_rows = list(csv.DictReader(f))

print(f"{len(sky_rows)} عکس آسمان واقعی")
count = 0
for row in sky_rows:
    img = cv2.imdecode(np.fromfile(str(REAL_SKY / "images" / row["file"]), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]
    excl = parse_exclude(row["exclude_boxes"], W, H)
    canvas = img.copy()
    lines, centers = [], []
    for _ in range(N_PER_SKY):
        d = int(round(np.exp(np.random.uniform(np.log(MIN_D), np.log(MAX_D)))))
        placed = False
        for _try in range(20):
            cx, cy = random.randint(3 * d, W - 3 * d), random.randint(3 * d, H - 3 * d)
            if in_excluded(cx, cy, d, excl):
                continue
            if any((cx - a) ** 2 + (cy - b) ** 2 < 60 ** 2 for a, b in centers):
                continue
            ok, sky_std = sky_ok(canvas, cx, cy, d)
            if ok:
                placed = True
                break
        if not placed:
            continue
        bgr, alpha = make_balloon(d)
        x1, y1, x2, y2 = paste(canvas, bgr, alpha, cx, cy, float(np.clip(sky_std * 0.5, 0.5, 4)))
        centers.append((cx, cy))
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    if not lines:
        continue
    name = f"syn2_{count:05d}"
    cv2.imwrite(str(OUT_IMG / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT_LBL / f"{name}.txt").write_text("\n".join(lines))
    count += 1
    if count % 1000 == 0:
        print(f"  {count} ساخته شد ...")

print(f"\n{count} عکس مصنوعی ساخته شد توی C:\\BalloonProject\\synthetic_v2")