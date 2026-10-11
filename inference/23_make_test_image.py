import random
from pathlib import Path

import cv2
import numpy as np

REAL_SKY = Path(r"C:\BalloonProject\real_sky\images")
OUT_DIR = Path(r"C:\BalloonProject\test_big")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 4056, 3040
FOCAL_PX = 3871       # فاصلهٔ کانونی به پیکسل (از محاسبهٔ لنز 6mm روی IMX477)
BALLOON_DIAM_M = 1.5  # قطر واقعی بالن

random.seed(42)
np.random.seed(42)


def make_sky_canvas():
    files = list(REAL_SKY.glob("*.jpg")) + list(REAL_SKY.glob("*.png"))
    src = cv2.imdecode(np.fromfile(str(random.choice(files)), np.uint8), cv2.IMREAD_COLOR)
    canvas = cv2.resize(src, (W, H), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)  # کمی محو تا بلوک‌شدگی resize کمتر دیده شود
    return canvas


def draw_balloon(d):
    size = d + 4
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    mask = (dist <= d / 2).astype(np.float32)
    hue = random.randint(0, 4)
    sat = random.randint(180, 255)
    val = np.clip(200 * (1 - 0.25 * dist / (d / 2)), 0, 255)
    hsv = np.stack([np.full_like(val, hue), np.full_like(val, sat), val], -1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    bgr = cv2.GaussianBlur(bgr, (0, 0), 0.6)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.4)
    return bgr, mask


def paste(canvas, bgr, mask, cx, cy):
    n = bgr.shape[0]
    x0, y0 = cx - n // 2, cy - n // 2
    roi = canvas[y0:y0 + n, x0:x0 + n].astype(np.float32)
    a = mask[..., None]
    canvas[y0:y0 + n, x0:x0 + n] = np.clip(bgr * a + roi * (1 - a), 0, 255).astype(np.uint8)


# فاصله‌های واقعی که می‌خواهیم تست کنیم (متر)
TEST_DISTANCES = [150, 250, 350, 500, 650, 800]

canvas = make_sky_canvas()
gt = []  # ground truth: cx, cy, d_px, distance_m

margin = 300
positions = []
for dist_m in TEST_DISTANCES:
    d_px = FOCAL_PX * BALLOON_DIAM_M / dist_m
    for _try in range(30):
        cx = random.randint(margin, W - margin)
        cy = random.randint(margin, H - margin)
        if all((cx - px) ** 2 + (cy - py) ** 2 > 250 ** 2 for px, py in positions):
            positions.append((cx, cy))
            break
    else:
        continue
    d_int = max(int(round(d_px)), 3)
    bgr, mask = draw_balloon(d_int)
    paste(canvas, bgr, mask, cx, cy)
    gt.append((cx, cy, d_int, dist_m))
    print(f"فاصلهٔ واقعی {dist_m}m -> قطر پیکسلی {d_px:.1f} -> رسم‌شده {d_int}px در ({cx},{cy})")

out_path = OUT_DIR / "test_scene_1.jpg"
cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

with (OUT_DIR / "test_scene_1_gt.txt").open("w") as f:
    for cx, cy, d, dist_m in gt:
        f.write(f"{cx},{cy},{d},{dist_m}\n")

print(f"\nعکس تست ساخته شد: {out_path}")
print(f"Ground truth ذخیره شد: {OUT_DIR / 'test_scene_1_gt.txt'}")