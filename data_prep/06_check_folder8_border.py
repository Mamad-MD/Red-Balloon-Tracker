import glob, os
import cv2, numpy as np

ROOT = r"C:\BalloonProject\dataset\rboflow\8"
OUT = r"C:\BalloonProject\review\folder_8_border.jpg"
T = 160
MARGIN = 0.01

def read_boxes(lf):
    out = []
    with open(lf, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            v = [float(x) for x in p[1:]]
            x, y, w, h = v[:4]
            out.append((x - w/2, y - h/2, x + w/2, y + h/2))
    return out

def find_image(lf):
    base = os.path.splitext(os.path.basename(lf))[0]
    imgdir = os.path.join(os.path.dirname(os.path.dirname(lf)), "images")
    g = glob.glob(os.path.join(imgdir, base + ".*"))
    return g[0] if g else None

tiles = []
for lf in glob.glob(os.path.join(ROOT, "**", "labels", "*.txt"), recursive=True):
    for b in read_boxes(lf):
        x1, y1, x2, y2 = b
        if not (x1 < MARGIN or y1 < MARGIN or x2 > 1 - MARGIN or y2 > 1 - MARGIN):
            continue
        ip = find_image(lf)
        if not ip:
            continue
        img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        px1, py1 = max(int(x1*W)-10, 0), max(int(y1*H)-10, 0)
        px2, py2 = min(int(x2*W)+10, W), min(int(y2*H)+10, H)
        crop = img[py1:py2, px1:px2]
        if crop.size == 0:
            continue
        tiles.append(cv2.resize(crop, (T, T)))
        if len(tiles) >= 60:
            break
    if len(tiles) >= 60:
        break

rows = [np.hstack(tiles[i:i+10]) for i in range(0, len(tiles) - len(tiles) % 10, 10)]
cv2.imwrite(OUT, np.vstack(rows))
print(f"{len(tiles)} تا نمونه ذخیره شد توی {OUT}")