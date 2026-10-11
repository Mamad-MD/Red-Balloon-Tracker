# 34_motion_filter_final.py
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\BalloonProject\runs\main\weights\best.pt"
VIDEO_DIR = Path(r"C:\BalloonProject\test_big")
VIDEOS = ["video_cloudy.mp4", "video_sunset.mp4", "video_windy_bg.mp4", "video_empty_sunset.mp4"]
TILE = 640
OVERLAP = 128
BLUR_SIGMA = 3.0
MOTION_THRESH = 45          # آستانه‌ی جدید (قبلا ۱۵ بود)
MOTION_MIN_PIXELS = 20
RED_MIN_PIXELS = 5
CONF = 0.25
IOU_NMS = 0.4


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


def nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= iou_thresh]
    return keep


model = YOLO(MODEL_PATH)
warm = np.zeros((TILE, TILE, 3), np.uint8)
_ = model.predict([warm] * 8, conf=CONF, verbose=False, device=0)

print(f"{'ویدیو':<25} | {'فریم':>5} | {'%تکه به مدل':>11} | {'FPS واقعی':>10} | {'میانگین تشخیص':>13}")

for vid_name in VIDEOS:
    path = VIDEO_DIR / vid_name
    if not path.exists():
        print(f"رد شد (پیدا نشد): {vid_name}")
        continue
    cap = cv2.VideoCapture(str(path))
    ok, prev = cap.read()
    H, W = prev.shape[:2]
    tiles = make_tiles(W, H, TILE, OVERLAP)
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (0, 0), BLUR_SIGMA)

    frame_idx = 0
    total_sent, total_dets = 0, 0
    frame_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t0 = time.time()

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), BLUR_SIGMA)
        diff = cv2.absdiff(gray, prev_gray)

        sel_tiles, sel_xy = [], []
        for x, y in tiles:
            d = diff[y:y + TILE, x:x + TILE]
            motion_px = int((d > MOTION_THRESH).sum())
            tb = frame[y:y + TILE, x:x + TILE]
            red_px = red_pixels(tb)
            if motion_px >= MOTION_MIN_PIXELS or red_px >= RED_MIN_PIXELS:
                sel_tiles.append(tb)
                sel_xy.append((x, y))
        total_sent += len(sel_tiles)

        boxes, scores = [], []
        if sel_tiles:
            results = model.predict(sel_tiles, conf=CONF, verbose=False, device=0)
            for (x, y), res in zip(sel_xy, results):
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                    scores.append(float(box.conf[0]))
        keep = nms(boxes, scores, IOU_NMS)
        total_dets += len(keep)

        frame_times.append(time.time() - t0)
        prev_gray = gray

    cap.release()
    avg_time = sum(frame_times) / len(frame_times)
    fps = 1 / avg_time
    pct = 100 * total_sent / (frame_idx * len(tiles))
    avg_dets = total_dets / frame_idx
    print(f"{vid_name:<25} | {frame_idx:>5} | {pct:>10.1f}% | {fps:>9.2f} | {avg_dets:>13.2f}")