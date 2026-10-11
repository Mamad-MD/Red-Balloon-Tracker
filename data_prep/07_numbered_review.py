import glob, os
import cv2, numpy as np

FOLDERS = ["45", "54", "67", "69", "70"]
ROOT = r"C:\BalloonProject\dataset\rboflow"
OUT = r"C:\BalloonProject\review"
T = 140

def read_boxes(lf):
    out = []
    with open(lf, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            v = [float(x) for x in p[1:]]
            if len(v) == 4:
                x, y, w, h = v
                out.append((x - w/2, y - h/2, x + w/2, y + h/2))
            elif len(v) >= 6:
                xs, ys = v[0::2], v[1::2]
                out.append((min(xs), min(ys), max(xs), max(ys)))
    return out

def find_image(lf):
    base = os.path.splitext(os.path.basename(lf))[0]
    imgdir = os.path.join(os.path.dirname(os.path.dirname(lf)), "images")
    g = glob.glob(os.path.join(imgdir, base + ".*"))
    return g[0] if g else None

for folder in FOLDERS:
    root = os.path.join(ROOT, folder)
    items = []
    for lf in glob.glob(os.path.join(root, "**", "labels", "*.txt"), recursive=True):
        for b in read_boxes(lf):
            items.append((lf, b))

    tiles = []
    for idx, (lf, b) in enumerate(items):
        ip = find_image(lf)
        if not ip:
            continue
        img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        x1, y1, x2, y2 = b
        px1, py1 = max(int(x1*W)-8, 0), max(int(y1*H)-8, 0)
        px2, py2 = min(int(x2*W)+8, W), min(int(y2*H)+8, H)
        crop = img[py1:py2, px1:px2]
        if crop.size == 0:
            continue
        tile = cv2.resize(crop, (T, T))
        cv2.putText(tile, str(idx), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        tiles.append(tile)

    per_row = 12
    pad = per_row - (len(tiles) % per_row or per_row)
    tiles += [np.zeros((T, T, 3), np.uint8)] * pad
    rows = [np.hstack(tiles[i:i+per_row]) for i in range(0, len(tiles), per_row)]
    grid = np.vstack(rows)
    out_path = os.path.join(OUT, f"folder_{folder}_numbered.jpg")
    cv2.imwrite(out_path, grid)
    print(f"پوشه {folder}: {len(items)} باکس -> {out_path}")