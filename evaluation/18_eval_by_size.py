from pathlib import Path
from ultralytics import YOLO
import numpy as np

MODEL = r"C:\BalloonProject\runs\main\weights\best.pt"
DATA = r"C:\BalloonProject\dataset_final\data.yaml"
IMG_DIR = Path(r"C:\BalloonProject\dataset_final\images\val")
LBL_DIR = Path(r"C:\BalloonProject\dataset_final\labels\val")
CONF = 0.25
IOU_MATCH = 0.3

model = YOLO(MODEL)

bins = {"tiny(<20px)": [0, 0], "small(20-32px)": [0, 0],
        "medium(32-64px)": [0, 0], "large(>64px)": [0, 0]}

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

def bin_of(d):
    if d < 20: return "tiny(<20px)"
    if d < 32: return "small(20-32px)"
    if d < 64: return "medium(32-64px)"
    return "large(>64px)"

files = sorted(IMG_DIR.iterdir())
for i, img_path in enumerate(files):
    lp = LBL_DIR / (img_path.stem + ".txt")
    gts = []
    if lp.exists():
        from PIL import Image
        with Image.open(img_path) as im:
            W, H = im.size
        gts = read_gt(lp, W, H)
    if not gts:
        continue
    res = model.predict(str(img_path), conf=CONF, verbose=False)[0]
    preds = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
    for cx, cy, w, h in gts:
        d = max(w, h)
        key = bin_of(d)
        bins[key][1] += 1
        gtbox = (cx, cy, w, h)
        if any(iou(gtbox, p) >= IOU_MATCH for p in preds):
            bins[key][0] += 1
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(files)}")

print("\nاندازه | Recall | تعداد بالن")
for k, (hit, tot) in bins.items():
    r = hit / tot if tot else 0
    print(f"{k:>16} | {r:.3f} | {tot}")