from pathlib import Path
import random

import cv2
import numpy as np

REAL_SKY = Path(r"C:\BalloonProject\real_sky\images")
OUT_DIR = Path(r"C:\BalloonProject\test_big")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 4056, 3040
N_FRAMES = 40
FPS_OUT = 10
BALLOON_D = 12

random.seed(1)
np.random.seed(1)


def load_random_sky():
    files = list(REAL_SKY.glob("*.jpg")) + list(REAL_SKY.glob("*.png"))
    for _ in range(20):
        src = cv2.imdecode(np.fromfile(str(random.choice(files)), np.uint8), cv2.IMREAD_COLOR)
        if src is not None:
            return cv2.resize(src, (W + 60, H + 60), interpolation=cv2.INTER_CUBIC)
    raise RuntimeError("عکس آسمان پیدا نشد")


def warm_tint(img, strength=0.25):
    img = img.astype(np.float32)
    img[..., 2] = np.clip(img[..., 2] * (1 + strength), 0, 255)   # قرمز بیشتر
    img[..., 0] = np.clip(img[..., 0] * (1 - strength * 0.5), 0, 255)  # آبی کمتر
    return img.astype(np.uint8)


def add_cloud_noise(img, strength=18):
    noise = np.random.normal(0, strength, img.shape[:2])
    blur = cv2.GaussianBlur(noise, (0, 0), 25)
    blur = (blur - blur.min()) / (blur.max() - blur.min() + 1e-6) * 60 - 30
    out = img.astype(np.float32) + blur[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


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


def make_video(name, tint=False, cloud=False, jitter=2, with_balloon=True, wind=0):
    base = load_random_sky()
    writer = cv2.VideoWriter(str(OUT_DIR / name), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H))
    bgr_b, mask_b = draw_balloon(BALLOON_D) if with_balloon else (None, None)
    cx, cy = 500, 1800
    vx, vy = 9.0, -1.5

    for i in range(N_FRAMES):
        ox, oy = random.randint(-jitter, jitter), random.randint(-jitter, jitter)
        wx = int(wind * i)  # حرکت آهستهٔ کلی پس‌زمینه (مثل باد بردن ابر)
        sx = 30 + ox + (wx % 40)
        sy = 30 + oy
        frame = base[sy:sy + H, sx:sx + W].copy()
        if tint:
            frame = warm_tint(frame)
        if cloud:
            frame = add_cloud_noise(frame)

        if with_balloon:
            n = bgr_b.shape[0]
            px, py = int(cx), int(cy)
            roi = frame[py:py + n, px:px + n].astype(np.float32)
            a = mask_b[..., None]
            frame[py:py + n, px:px + n] = np.clip(bgr_b * a + roi * (1 - a), 0, 255).astype(np.uint8)
            cx += vx
            cy += vy

        writer.write(frame)
    writer.release()
    print(f"ساخته شد: {name}")


make_video("video_cloudy.mp4", tint=False, cloud=True, jitter=3, with_balloon=True)
make_video("video_sunset.mp4", tint=True, cloud=False, jitter=2, with_balloon=True)
make_video("video_windy_bg.mp4", tint=False, cloud=True, jitter=4, with_balloon=True, wind=3)
make_video("video_empty_sunset.mp4", tint=True, cloud=True, jitter=3, with_balloon=False)

print("\nهر چهار ویدیو در C:\\BalloonProject\\test_big ساخته شدند")