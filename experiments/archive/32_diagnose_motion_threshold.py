# 32_diagnose_motion_threshold.py
from pathlib import Path
import cv2
import numpy as np

VIDEO_DIR = Path(r"C:\BalloonProject\test_big")
VIDEOS = ["video_cloudy.mp4", "video_windy_bg.mp4", "video_empty_sunset.mp4", "video_sunset.mp4"]
TILE = 640
OVERLAP = 128
MOTION_THRESH = 15


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
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    all_motion_px = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        for x, y in tiles:
            d = diff[y:y + TILE, x:x + TILE]
            motion_px = int((d > MOTION_THRESH).sum())
            all_motion_px.append(motion_px)
        prev_gray = gray

    arr = np.array(all_motion_px)
    print(f"\n=== {vid_name} ===")
    print(f"تعداد تکه بررسی‌شده: {len(arr)} (در {frame_idx} فریم)")
    print(f"میانگین پیکسل متحرک: {arr.mean():.0f}")
    print(f"میانه: {np.median(arr):.0f}")
    print(f"صدک ۵۰: {np.percentile(arr,50):.0f} | صدک ۷۵: {np.percentile(arr,75):.0f} | "
          f"صدک ۹۰: {np.percentile(arr,90):.0f} | صدک ۹۹: {np.percentile(arr,99):.0f} | بیشینه: {arr.max():.0f}")
    print(f"درصد تکه‌هایی که با آستانه‌ی فعلی (۲۰ پیکسل) رد می‌شوند: "
          f"{100*(arr >= 20).mean():.1f}%")