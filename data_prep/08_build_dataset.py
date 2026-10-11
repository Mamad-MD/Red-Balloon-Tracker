import glob, os, shutil, csv
import cv2
import numpy as np

ROOT = r"C:\BalloonProject\dataset\rboflow"
OUT = r"C:\BalloonProject\merged_raw"
BORDER_MARGIN = 0.01
LOOSE_THRESH = 0.3

# شماره‌ی باکس‌هایی که با بررسی چشمی (کانتکت‌شیت شماره‌دار) تصمیم به حذفشون گرفتیم
REMOVE_27 = set(range(19, 91)) | {0, 1, 2, 3, 5, 95, 96, 100, 101, 102, 130, 134, 156, 177}
REMOVE_32 = set(range(9, 33)) | {35, 43, 46, 47, 48, 63}

# چه پوشه‌ای، چه کلاس‌هاش نگه داشته میشه (فقط کلاس‌های لیست‌شده تبدیل به red_balloon میشن، بقیه کلاس‌ها کامل نادیده گرفته میشن)
FOLDER_CLASSES = {
    "8": None,          # None یعنی همه‌ی کلاس‌ها (فقط یک کلاس داره)
    "11": None,
    "15": {0, 1},        # کلاس 2 (آبی) نادیده گرفته میشه
    "27": None,
    "32": None,
    "45": None,
    "47": {4},
    "50": None,
    "52": None, "53": None, "55": None,
    "68": None, "71": None,
}

# پوشه‌هایی که باکس بی‌رنگ (بدون هیچ پیکسل قرمز) حذف بشه
FILTER_NO_RED = {"11", "15", "50", "45"}
# پوشه‌هایی که باکس لمس‌مرز حذف بشه
FILTER_BORDER = {"8", "47"}
# پوشه‌هایی که باکس شل باید تنگ‌تر بشه
FILTER_TIGHTEN = {"45", "70"}
# آستانه‌ی کشیدگی غیرعادی برای پوشه 50
ASPECT_MAX = 3.0


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)


def red_extent(img, x1, y1, x2, y2):
    """محدوده‌ی واقعی پیکسل قرمز داخل کراپ رو برمی‌گردونه، یا None اگه قرمزی نبود."""
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    mask = red_mask(crop)
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def read_boxes(lf):
    out = []
    with open(lf, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            v = [float(x) for x in p[1:]]
            if len(v) == 4:
                x, y, w, h = v
                out.append((cls, x - w/2, y - h/2, x + w/2, y + h/2))
            elif len(v) >= 6 and len(v) % 2 == 0:
                xs, ys = v[0::2], v[1::2]
                out.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return out


def find_image(lf):
    base = os.path.splitext(os.path.basename(lf))[0]
    imgdir = os.path.join(os.path.dirname(os.path.dirname(lf)), "images")
    g = glob.glob(os.path.join(imgdir, base + ".*"))
    return g[0] if g else None


os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)
manifest = []
img_counter = 0

for folder, keep_classes in FOLDER_CLASSES.items():
    root = os.path.join(ROOT, folder)
    box_counter = -1  # فقط برای 27 و 32 استفاده میشه، شمارش سراسری باکس‌های این پوشه
    for split in ("train", "valid", "test"):
        lbl_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for lf in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
            boxes = read_boxes(lf)
            ip = find_image(lf)
            if not ip:
                continue
            img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]

            kept_lines = []
            for (cls, x1, y1, x2, y2) in boxes:
                box_counter += 1
                if keep_classes is not None and cls not in keep_classes:
                    continue

                if folder in {"27", "32"}:
                    idx_set = REMOVE_27 if folder == "27" else REMOVE_32
                    if box_counter in idx_set:
                        continue

                touches_border = (x1 < BORDER_MARGIN or y1 < BORDER_MARGIN or
                                   x2 > 1 - BORDER_MARGIN or y2 > 1 - BORDER_MARGIN)
                if folder in FILTER_BORDER and touches_border:
                    continue

                px1, py1 = max(int(x1*W), 0), max(int(y1*H), 0)
                px2, py2 = min(int(x2*W), W), min(int(y2*H), H)
                ext = red_extent(img, px1, py1, px2, py2)

                if folder in FILTER_NO_RED and ext is None:
                    continue

                if folder == "50":
                    bw, bh = (x2-x1)*W, (y2-y1)*H
                    aspect = bw / max(bh, 1e-6)
                    if ext is None or aspect > ASPECT_MAX or aspect < 1/ASPECT_MAX:
                        continue

                if folder in FILTER_TIGHTEN and ext is not None:
                    box_area = (px2-px1) * (py2-py1)
                    ex1, ey1, ex2, ey2 = ext
                    red_area = (ex2-ex1) * (ey2-ey1)
                    if box_area > 0 and red_area / box_area < LOOSE_THRESH:
                        pad = 2
                        nx1 = max(px1 + ex1 - pad, 0)
                        ny1 = max(py1 + ey1 - pad, 0)
                        nx2 = min(px1 + ex2 + pad, W)
                        ny2 = min(py1 + ey2 + pad, H)
                        x1, y1, x2, y2 = nx1/W, ny1/H, nx2/W, ny2/H

                cx, cy = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1, y2-y1
                kept_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # حتی اگه هیچ باکسی نموند، عکس رو به عنوان پس‌زمینه نگه می‌داریم
            img_counter += 1
            new_name = f"{folder}_{img_counter:06d}"
            ext_name = os.path.splitext(ip)[1]
            shutil.copy(ip, os.path.join(OUT, "images", new_name + ext_name))
            with open(os.path.join(OUT, "labels", new_name + ".txt"), "w") as f:
                f.write("\n".join(kept_lines))
            manifest.append({
                "new_name": new_name, "source_folder": folder,
                "source_split": split, "source_file": os.path.basename(ip),
                "n_boxes": len(kept_lines),
            })
    print(f"پوشه {folder} تمام شد")

with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=manifest[0].keys())
    wr.writeheader()
    wr.writerows(manifest)

total_boxes = sum(m["n_boxes"] for m in manifest)
empty = sum(1 for m in manifest if m["n_boxes"] == 0)
print(f"\nجمع کل: {len(manifest)} عکس، {total_boxes} باکس، {empty} عکس بدون باکس (پس‌زمینه)")