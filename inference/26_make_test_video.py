from pathlib import Path
import random

import cv2
import numpy as np

REAL_SKY = Path(r"C:\BalloonProject\real_sky\images")
OUT_PATH = Path(r"C:\BalloonProject\test_big\test_video.mp4")

W, H = 4056, 3040
N_FRAMES = 40
FPS_OUT = 10
BALLOON_D = 12  # حدود همان اندازهٔ هدف در 500 متری

random.seed(7)
np.random.seed(7)


def load_sky():
    files = list(REAL_SKY.glob("*.jpg")) + list(REAL_SKY.glob("*.png"))
    src = cv2.imdecode(np.fromfile(str(random.choice(files)), np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(src, (W + 40, H + 40), interpolation=cv2.INTER_CUBIC)


def draw_balloon(d):
    size = d + 4
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    mask = (dist <= d / 2).astype(np.float32)
    hsv = np.stack([np.full_like(mask, 2), np.full_like(mask, 220),
                     np.clip(200 * (1 - 0.2 * dist / (d / 2)), 0, 255)], -1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    return cv2.GaussianBlur(bgr, (0, 0), 0.5), cv2.GaussianBlur(mask, (0, 0), 0.4)


sky_big = load_sky()
bgr_b, mask_b = draw_balloon(BALLOON_D)

writer = cv2.VideoWriter(str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H))

cx, cy = 500, 1800
vx, vy = 9.0, -1.5  # سرعت حرکت بالن روی صحنه (پیکسل بر فریم)

for i in range(N_FRAMES):
    # جیتر خیلی کوچک دوربین (شبیه لرزش نصب، نه چرخش واقعی)
    ox, oy = random.randint(-2, 2), random.randint(-2, 2)
    frame = sky_big[20 + oy:20 + oy + H, 20 + ox:20 + ox + W].copy()
    frame = np.clip(frame.astype(np.int16) + np.random.randint(-3, 3, frame.shape), 0, 255).astype(np.uint8)

    n = bgr_b.shape[0]
    px, py = int(cx), int(cy)
    roi = frame[py:py + n, px:px + n].astype(np.float32)
    a = mask_b[..., None]
    frame[py:py + n, px:px + n] = np.clip(bgr_b * a + roi * (1 - a), 0, 255).astype(np.uint8)

    writer.write(frame)
    cx += vx
    cy += vy

writer.release()
print(f"ویدیوی تست ساخته شد: {OUT_PATH} ({N_FRAMES} فریم، {FPS_OUT} fps)")
print(f"بالن با اندازهٔ {BALLOON_D}px از ({500},{1800}) با سرعت ({vx},{vy}) پیکسل/فریم حرکت می‌کند")