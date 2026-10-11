from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\BalloonProject\runs\main\weights\best.pt"
VIDEO_DIR = Path(r"C:\BalloonProject\test_big")
VIDEOS = ["video_cloudy.mp4", "video_sunset.mp4", "video_windy_bg.mp4", "video_empty_sunset.mp4"]
TILE = 640
OVERLAP = 128
MOTION_THRESH = 15
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

for vid_name in VIDEOS:
    path = VIDEO_DIR / vid_name
    if not path.exists():
        print(f"رد شد (پیدا نشد): {vid_name}")
        continue
    cap = cv2.VideoCapture(str(path))
    ok, prev = cap.read()
    H, W = prev.shape[:2]
    tiles = make_tiles(W, H, TILE, OVERLAP)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    total_all, total_sent, total_final_dets = 0, 0, 0
    max_dets_in_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)

        sel_tiles, sel_xy = [], []
        for x, y in tiles:
            total_all += 1
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
        total_final_dets += len(keep)
        max_dets_in_frame = max(max_dets_in_frame, len(keep))
        prev_gray = gray

    pct = 100 * total_sent / total_all
    avg_dets = total_final_dets / frame_idx
    print(f"\n=== {vid_name} ===")
    print(f"فریم: {frame_idx} | تکه به مدل: {pct:.1f}٪ | میانگین تشخیص در فریم: {avg_dets:.2f} | حداکثر در یک فریم: {max_dets_in_frame}")
    cap.release()