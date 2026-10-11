import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = r"C:\BalloonProject\test_big\test_video.mp4"
MODEL_PATH = r"C:\BalloonProject\runs\main\weights\best.pt"
TILE = 640
OVERLAP = 128
MOTION_THRESH = 15       # اختلاف پیکسلی برای اعلام «حرکت» (0-255)
MOTION_MIN_PIXELS = 20   # حداقل تعداد پیکسل متحرک برای علامت‌زدن تکه
RED_MIN_PIXELS = 5       # حداقل پیکسل قرمز مشکوک برای علامت‌زدن تکه
CONF = 0.25


def make_tiles(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < W:
        xs.append(W - tile)
    if not ys or ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


def red_pixels(tile_bgr):
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)
    return int(mask.sum())


cap = cv2.VideoCapture(VIDEO_PATH)
ok, prev = cap.read()
H, W = prev.shape[:2]
tiles = make_tiles(W, H, TILE, OVERLAP)
print(f"{len(tiles)} تکه در هر فریم، {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} فریم کل")

model = YOLO(MODEL_PATH)

frame_idx = 0
total_tiles_all = 0
total_tiles_sent = 0

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)

    selected_tiles, selected_coords = [], []
    for x, y in tiles:
        total_tiles_all += 1
        d = diff[y:y + TILE, x:x + TILE]
        motion_px = int((d > MOTION_THRESH).sum())
        tile_bgr = frame[y:y + TILE, x:x + TILE]
        red_px = red_pixels(tile_bgr)
        if motion_px >= MOTION_MIN_PIXELS or red_px >= RED_MIN_PIXELS:
            selected_tiles.append(tile_bgr)
            selected_coords.append((x, y))

    total_tiles_sent += len(selected_tiles)

    n_det = 0
    if selected_tiles:
        results = model.predict(selected_tiles, conf=CONF, verbose=False, device=0)
        n_det = sum(len(r.boxes) for r in results)

    print(f"فریم {frame_idx}: {len(selected_tiles)}/{len(tiles)} تکه به مدل رفت | تشخیص خام: {n_det}")
    prev_gray = gray

cap.release()
pct = 100 * total_tiles_sent / total_tiles_all
print(f"\nجمع کل: {total_tiles_sent} از {total_tiles_all} تکه به مدل داده شد ({pct:.1f}٪)")
print(f"یعنی حدود {100/pct*1:.1f} برابر کمتر از حالت کامل (48 تکه در هر فریم)")