from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

MODEL = r"C:\BalloonProject\runs\main\weights\best.pt"
IMG_DIR = Path(r"C:\BalloonProject\dataset_final\images\val")
LBL_DIR = Path(r"C:\BalloonProject\dataset_final\labels\val")
OUT = Path(r"C:\BalloonProject\review\misses.jpg")
CONF = 0.25
IOU_MATCH = 0.3
MAX_SIZE = 36  # فقط بالن‌های ریز و کوچک را بررسی کن
T = 160

model = YOLO(MODEL)

def read_gt(lp, W, H):
    out = []
    for line in lp.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        out.append((cx * W, cy * H, w * W, h * H))
    return out

def iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1[0]-b1[2]/2, b1[1]-b1[3]/2, b1[0]+b1[2]/2, b1[1]+b1[3]/2
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    a1 = (ax2-ax1)*(ay2-ay1)
    a2 = (bx2-bx1)*(by2-by1)
    return inter / max(a1 + a2 - inter, 1e-6)

tiles = []
files = sorted(IMG_DIR.iterdir())
for img_path in files:
    lp = LBL_DIR / (img_path.stem + ".txt")
    if not lp.exists():
        continue
    with Image.open(img_path) as im:
        W, H = im.size
    gts = read_gt(lp, W, H)
    if not gts:
        continue
    res = model.predict(str(img_path), conf=CONF, verbose=False)[0]
    preds = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
    img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
    for cx, cy, w, h in gts:
        d = max(w, h)
        if d > MAX_SIZE:
            continue
        gtbox = (cx, cy, w, h)
        hit = any(iou(gtbox, p) >= IOU_MATCH for p in preds)
        if hit:
            continue
        half = int(max(d, 12) * 2)
        pad = half + 2
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40,40,40))
        px, py = int(cx) + pad, int(cy) + pad
        crop = big[py-half:py+half, px-half:px+half]
        crop = cv2.resize(crop, (T, T), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"d={d:.0f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        tiles.append(crop)

print(f"{len(tiles)} بالن ریز/کوچک از دست‌رفته پیدا شد")
if tiles:
    per_row = 10
    pad_n = per_row - (len(tiles) % per_row or per_row)
    tiles += [np.zeros((T, T, 3), np.uint8)] * pad_n
    rows = [np.hstack(tiles[i:i+per_row]) for i in range(0, len(tiles), per_row)]
    cv2.imwrite(str(OUT), np.vstack(rows))
    print(f"ذخیره شد: {OUT}")