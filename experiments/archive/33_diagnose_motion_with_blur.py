# 33_diagnose_motion_with_blur.py
from pathlib import Path
import cv2
import numpy as np

VIDEO_DIR = Path(r"C:\BalloonProject\test_big")
VIDEOS = ["video_cloudy.mp4", "video_windy_bg.mp4", "video_empty_sunset.mp4", "video_sunset.mp4"]
TILE = 640
OVERLAP = 128
BLUR_SIGMA = 3.0  # هموارسازی قبل از مقایسه
THRESHOLDS_TO_TEST = [15, 25, 35, 45, 60]


def make_tiles(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < W:
        xs.append(W - tile)
    if not ys or ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


for vid_name in VIDEOS:
    path = VIDEO_DIR / vid_name
    cap = cv2.VideoCapture(str(path))
    ok, prev = cap.read()
    H, W = prev.shape[:2]
    tiles = make_tiles(W, H, TILE, OVERLAP)
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (0, 0), BLUR_SIGMA)

    all_motion_px = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), BLUR_SIGMA)
        diff = cv2.absdiff(gray, prev_gray)
        for x, y in tiles:
            d = diff[y:y + TILE, x:x + TILE]
            all_motion_px.append(d)
        prev_gray = gray

    print(f"\n=== {vid_name} (بعد از بلور سیگما={BLUR_SIGMA}) ===")
    for thresh in THRESHOLDS_TO_TEST:
        counts = [int((d > thresh).sum()) for d in all_motion_px]
        arr = np.array(counts)
        print(f"  آستانه‌ی شدت={thresh:>3} -> میانگین پیکسل عبورکرده: {arr.mean():>8.0f} | "
              f"صدک ۹۰: {np.percentile(arr,90):>7.0f} | بیشینه: {arr.max():>8.0f} | "
              f"درصد تکه‌های رد از فیلتر (>=20px): {100*(arr>=20).mean():>5.1f}%")