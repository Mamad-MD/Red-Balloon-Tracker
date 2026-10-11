import glob, os
import cv2, numpy as np

ROOT = r"C:\BalloonProject\dataset\rboflow\8"
OUT = r"C:\BalloonProject\review\folder_8_empty.jpg"
T = 160
N = 60

tiles = []
for lbl_dir in glob.glob(os.path.join(ROOT, "*", "labels")):
    img_dir = os.path.join(os.path.dirname(lbl_dir), "images")
    for lf in glob.glob(os.path.join(lbl_dir, "*.txt")):
        content = open(lf, encoding="utf-8", errors="ignore").read().strip()
        if content:
            continue
        base = os.path.splitext(os.path.basename(lf))[0]
        g = glob.glob(os.path.join(img_dir, base + ".*"))
        if not g:
            continue
        img = cv2.imdecode(np.fromfile(g[0], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tiles.append(cv2.resize(img, (T, T)))
        if len(tiles) >= N:
            break
    if len(tiles) >= N:
        break

per_row = 10
pad = per_row - (len(tiles) % per_row or per_row)
tiles += [np.zeros((T, T, 3), np.uint8)] * pad
rows = [np.hstack(tiles[i:i+per_row]) for i in range(0, len(tiles), per_row)]
cv2.imwrite(OUT, np.vstack(rows))
print(f"{len(tiles)} نمونه ذخیره شد")