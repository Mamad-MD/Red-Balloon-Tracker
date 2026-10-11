import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\BalloonProject\runs\main\weights\best.pt"
IMG_PATH = r"C:\BalloonProject\test_big\test_scene_1.jpg"
GT_PATH = r"C:\BalloonProject\test_big\test_scene_1_gt.txt"

TILE = 640
OVERLAP = 128          # هم‌پوشانی بین تکه‌ها (پیکسل)
CONF = 0.25
IOU_NMS = 0.4
FOCAL_PX = 3871
BALLOON_DIAM_M = 1.5
DIST_ACCURACY_PCT = 15  # عدم قطعیت تخمینی فاصله (بدون کالیبراسیون واقعی لنز)


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

def estimate_distance(d_px):
    if d_px <= 0:
        return None
    d_corrected = (d_px - 1.41) / 1.025   # اصلاح خطای سیستماتیک اندازه‌ی کادر
    if d_corrected <= 0:
        return None
    return FOCAL_PX * BALLOON_DIAM_M / d_corrected


def load_gt(path):
    gt = []
    for line in Path(path).read_text().splitlines():
        cx, cy, d, dist_m = line.split(",")
        gt.append((float(cx), float(cy), float(d), float(dist_m)))
    return gt


model = YOLO(MODEL_PATH)

img = cv2.imdecode(np.fromfile(IMG_PATH, np.uint8), cv2.IMREAD_COLOR)
H, W = img.shape[:2]
tiles = make_tiles(W, H, TILE, OVERLAP)
print(f"عکس {W}x{H} به {len(tiles)} تکه بریده شد")

t0 = time.time()
tile_imgs = [img[y:y + TILE, x:x + TILE] for x, y in tiles]
results = model.predict(tile_imgs, conf=CONF, verbose=False, device=0)
t_infer = time.time() - t0

all_boxes, all_scores = [], []
for (x, y), res in zip(tiles, results):
    for box in res.boxes:
        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
        all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
        all_scores.append(float(box.conf[0]))

keep = nms(all_boxes, all_scores, IOU_NMS)
detections = [(all_boxes[i], all_scores[i]) for i in keep]

print(f"\nزمان استنتاج (فقط مدل، بدون خواندن/برش عکس): {t_infer:.3f} ثانیه برای {len(tiles)} تکه")
print(f"تعداد بالن تشخیص‌داده‌شده بعد از ادغام: {len(detections)}")

print("\nموقعیت(x,y) | قطر(px) | فاصلهٔ تخمینی(m) | اطمینان(conf)")
results_out = []
for box, score in sorted(detections, key=lambda d: -d[1]):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    d_px = max(x2 - x1, y2 - y1)
    dist = estimate_distance(d_px)
    results_out.append((cx, cy, d_px, dist, score))
    print(f"({cx:.0f},{cy:.0f}) | {d_px:.1f} | {dist:.0f} ± {dist*DIST_ACCURACY_PCT/100:.0f} | {score:.2f}")

# مقایسه با ground truth (چون عکس تست است و فاصلهٔ واقعی را می‌دانیم)
gt = load_gt(GT_PATH)
print("\n--- مقایسه با فاصلهٔ واقعی (فقط چون این عکس تست است) ---")
print("فاصلهٔ واقعی(m) | قطر واقعی(px) | پیدا شد؟ | فاصلهٔ تخمینی(m) | خطا")
for gcx, gcy, gd, gdist in gt:
    best = None
    for cx, cy, d_px, dist, score in results_out:
        if (cx - gcx) ** 2 + (cy - gcy) ** 2 < (gd * 3) ** 2:
            best = (cx, cy, d_px, dist, score)
            break
    if best:
        _, _, d_px, dist, score = best
        err = 100 * (dist - gdist) / gdist
        print(f"{gdist:.0f} | {gd:.0f} | بله | {dist:.0f} | {err:+.1f}%")
    else:
        print(f"{gdist:.0f} | {gd:.0f} | نه (از دست رفت) | - | -")