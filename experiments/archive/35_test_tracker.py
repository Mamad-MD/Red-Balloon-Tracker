# 35_test_tracker.py
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\BalloonProject\runs\main\weights\best.pt"
VIDEO_PATH = r"C:\BalloonProject\test_big\video_sunset.mp4"
OUT_VIDEO = r"C:\BalloonProject\test_big\video_sunset_tracked.mp4"
TILE = 640
OVERLAP = 128
CONF = 0.25
IOU_NMS = 0.4
MAX_MATCH_DIST = 80     # حداکثر فاصله‌ی پیکسلی برای اینکه دو تشخیص «همان یک بالن» شمرده شوند
MAX_MISSED_FRAMES = 5   # اگر بالن این‌قدر فریم پیاپی دیده نشد، ID‌اش حذف می‌شود


def make_tiles(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < W:
        xs.append(W - tile)
    if not ys or ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


def nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    idxs = scores_arr.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes_arr[i, 0], boxes_arr[rest, 0])
        yy1 = np.maximum(boxes_arr[i, 1], boxes_arr[rest, 1])
        xx2 = np.minimum(boxes_arr[i, 2], boxes_arr[rest, 2])
        yy2 = np.minimum(boxes_arr[i, 3], boxes_arr[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes_arr[i, 2] - boxes_arr[i, 0]) * (boxes_arr[i, 3] - boxes_arr[i, 1])
        area_r = (boxes_arr[rest, 2] - boxes_arr[rest, 0]) * (boxes_arr[rest, 3] - boxes_arr[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= iou_thresh]
    return keep


class CentroidTracker:
    """ردیاب ساده: هر بالن را با نزدیک‌ترین بالن فریم قبلی تطبیق می‌دهد."""
    def __init__(self, max_dist, max_missed):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}  # id -> {"center": (x,y), "missed": int, "history": [(x,y),...]}

    def update(self, detections):
        """detections: لیست (cx, cy, box). خروجی: لیست (track_id, cx, cy, box)."""
        unmatched_dets = list(range(len(detections)))
        matched_ids = set()

        # مرحله ۱: تطبیق هر ردِ فعال با نزدیک‌ترین تشخیص فریم فعلی
        for tid, track in list(self.tracks.items()):
            best_j, best_dist = None, self.max_dist
            for j in unmatched_dets:
                cx, cy, _ = detections[j]
                tx, ty = track["center"]
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                cx, cy, box = detections[best_j]
                track["center"] = (cx, cy)
                track["missed"] = 0
                track["history"].append((cx, cy))
                if len(track["history"]) > 30:
                    track["history"].pop(0)
                unmatched_dets.remove(best_j)
                matched_ids.add(tid)
            else:
                track["missed"] += 1

        # مرحله ۲: حذف ردهایی که خیلی فریم پیاپی دیده نشدند
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        # مرحله ۳: تشخیص‌های بدون تطبیق -> ID تازه
        for j in unmatched_dets:
            cx, cy, box = detections[j]
            self.tracks[self.next_id] = {"center": (cx, cy), "missed": 0, "history": [(cx, cy)]}
            matched_ids.add(self.next_id)
            self.next_id += 1

        # خروجی: فقط ردهایی که همین فریم آپدیت شدند
        out = []
        for tid in matched_ids:
            if self.tracks[tid]["missed"] == 0:
                cx, cy = self.tracks[tid]["center"]
                # پیدا کردن باکس متناظر
                for d_cx, d_cy, box in detections:
                    if abs(d_cx - cx) < 1e-3 and abs(d_cy - cy) < 1e-3:
                        out.append((tid, cx, cy, box, self.tracks[tid]["history"]))
                        break
        return out


model = YOLO(MODEL_PATH)
tracker = CentroidTracker(MAX_MATCH_DIST, MAX_MISSED_FRAMES)

cap = cv2.VideoCapture(VIDEO_PATH)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
tiles = make_tiles(W, H, TILE, OVERLAP)

writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
colors = {}

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    tile_imgs = [frame[y:y + TILE, x:x + TILE] for x, y in tiles]
    results = model.predict(tile_imgs, conf=CONF, verbose=False, device=0)

    all_boxes, all_scores = [], []
    for (x, y), res in zip(tiles, results):
        for box in res.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
            all_scores.append(float(box.conf[0]))

    keep = nms(all_boxes, all_scores, IOU_NMS)
    detections = []
    for i in keep:
        x1, y1, x2, y2 = all_boxes[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append((cx, cy, (x1, y1, x2, y2)))

    tracked = tracker.update(detections)

    vis = frame.copy()
    for tid, cx, cy, (x1, y1, x2, y2), history in tracked:
        if tid not in colors:
            rng = np.random.default_rng(tid)
            colors[tid] = tuple(int(c) for c in rng.integers(80, 255, 3))
        color = colors[tid]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"ID {tid}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for k in range(1, len(history)):
            cv2.line(vis, (int(history[k-1][0]), int(history[k-1][1])),
                      (int(history[k][0]), int(history[k][1])), color, 2)

    writer.write(vis)
    print(f"فریم {frame_idx}: {len(tracked)} بالن ردیابی شد -> ID ها: {[t[0] for t in tracked]}")

writer.release()
cap.release()
print(f"\nذخیره شد: {OUT_VIDEO}")