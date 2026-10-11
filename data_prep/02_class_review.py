import os, glob, random, csv
import cv2, numpy as np, yaml

ROOT = r"C:\BalloonProject\dataset\rboflow"
OUT = r"C:\BalloonProject\review"
os.makedirs(OUT, exist_ok=True)
random.seed(0)
N_SHOW, N_COLOR, T = 12, 200, 128

def sort_key(n):
    return (0, int(n)) if n.isdigit() else (1, n)

def load_names(d):
    p = glob.glob(os.path.join(d, "**", "data.yaml"), recursive=True)
    if not p:
        return []
    with open(p[0], encoding="utf-8") as f:
        n = yaml.safe_load(f).get("names", [])
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)

def read_boxes(lf):
    out = []
    with open(lf, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            v = [float(x) for x in p[1:]]
            if len(v) == 4:
                x, y, w, h = v
                out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
            elif len(v) >= 6 and len(v) % 2 == 0:
                xs, ys = v[0::2], v[1::2]
                out.append((c, min(xs), min(ys), max(xs), max(ys)))
    return out

def find_image(lf):
    base = os.path.splitext(os.path.basename(lf))[0]
    imgdir = os.path.join(os.path.dirname(os.path.dirname(lf)), "images")
    g = glob.glob(os.path.join(imgdir, base + ".*"))
    return g[0] if g else None

def red_fraction(img, b):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = b
    a, t = max(int(x1 * W), 0), max(int(y1 * H), 0)
    c, d = min(int(x2 * W) + 1, W), min(int(y2 * H) + 1, H)
    if c - a < 2 or d - t < 2:
        return None
    hsv = cv2.cvtColor(img[t:d, a:c], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)).mean())

def make_tile(img, b):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = b
    half = int(max((x2 - x1) * W, (y2 - y1) * H, 12) * 1.25)
    p = half + 2
    big = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cx, cy = int((x1 + x2) / 2 * W) + p, int((y1 + y2) / 2 * H) + p
    crop = cv2.resize(big[cy - half:cy + half, cx - half:cx + half], (T, T))
    sc = T / (2 * half)
    bw, bh = (x2 - x1) * W * sc, (y2 - y1) * H * sc
    cv2.rectangle(crop, (int(T / 2 - bw / 2), int(T / 2 - bh / 2)),
                  (int(T / 2 + bw / 2), int(T / 2 + bh / 2)), (0, 255, 0), 1)
    return crop

def header(txt, width):
    h = np.full((26, width, 3), 60, np.uint8)
    cv2.putText(h, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return h

rows = []
for name in sorted(os.listdir(ROOT), key=sort_key):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        continue
    names = load_names(d)
    per_class = {}
    for lf in glob.glob(os.path.join(d, "**", "labels", "*.txt"), recursive=True):
        for b in read_boxes(lf):
            per_class.setdefault(b[0], []).append((lf, b[1:]))
    strips = []
    for c in sorted(per_class):
        items = per_class[c]
        cname = str(names[c]) if c < len(names) else "?"
        areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for _, b in items)
        med = areas[len(areas) // 2] * 100
        reds, tiles = [], []
        for lf, b in random.sample(items, min(N_COLOR, len(items))):
            try:
                ip = find_image(lf)
                img = cv2.imread(ip) if ip else None
                if img is None:
                    continue
                r = red_fraction(img, b)
                if r is not None:
                    reds.append(r)
                if len(tiles) < N_SHOW:
                    tiles.append(make_tile(img, b))
            except Exception:
                continue
        rmed = float(np.median(reds)) * 100 if reds else -1
        r25 = 100 * sum(r >= 0.25 for r in reds) / len(reds) if reds else -1
        print(f"{name:>3} | class {c} '{cname[:25]}' | boxes={len(items):5d} | "
              f"median_area={med:6.2f}% | red_median={rmed:3.0f}% | red>=25%: {r25:3.0f}%")
        rows.append([name, c, cname, len(items), round(med, 2), round(rmed), round(r25)])
        txt = f"folder {name} | class {c} '{cname}' | boxes={len(items)} | red={rmed:.0f}%"
        txt = txt.encode("ascii", "replace").decode()
        blank = np.zeros((T, T, 3), np.uint8)
        row = np.hstack(tiles + [blank] * (N_SHOW - len(tiles)))
        strips += [header(txt, T * N_SHOW), row]
    if strips:
        cv2.imwrite(os.path.join(OUT, f"folder_{name}.jpg"), np.vstack(strips))

with open(r"C:\BalloonProject\class_review.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["folder", "class", "name", "boxes", "median_area_pct", "red_median_pct", "red25_pct"])
    w.writerows(rows)
print("\nDone. Sheets are in C:\\BalloonProject\\review")