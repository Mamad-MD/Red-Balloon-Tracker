import hashlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\BalloonProject\dataset\rboflow")
PAIRS = [("51", "61"), ("51", "60"), ("61", "60"), ("40", "41")]
NEAR_THRESHOLD = 10   # از ۲۵۶ بیت، حداکثر این تعداد اختلاف یعنی «تقریباً یکی»
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    base = ROOT / folder
    files = []
    for split in ("train", "valid", "test"):
        d = base / split / "images"
        if d.exists():
            files += [p for p in d.iterdir() if p.suffix.lower() in EXTS]
    return files


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()  # 256 بیت


cache = {}


def load(folder):
    if folder in cache:
        return cache[folder]
    files = list_images(folder)
    md5s, hashes, names = [], [], []
    for i, p in enumerate(files):
        h = dhash(p)
        if h is None:
            continue
        md5s.append(md5_of(p))
        hashes.append(h)
        names.append(p.name)
        if (i + 1) % 500 == 0:
            print(f"  پوشه {folder}: {i + 1}/{len(files)}")
    cache[folder] = (md5s, np.array(hashes), names)
    return cache[folder]


for a, b in PAIRS:
    print(f"\n=== مقایسه پوشه {a} با پوشه {b} ===")
    ma, ha, na = load(a)
    mb, hb, nb = load(b)
    exact = len(set(ma) & set(mb))

    near_count = 0
    examples = []
    for i in range(len(ha)):
        dist = (hb != ha[i]).sum(axis=1)
        j = int(dist.argmin())
        if dist[j] <= NEAR_THRESHOLD:
            near_count += 1
            if len(examples) < 3:
                examples.append((na[i], nb[j], int(dist[j])))

    print(f"تعداد عکس در {a}: {len(ha)} | در {b}: {len(hb)}")
    print(f"فایل کاملاً یکسان (MD5): {exact}")
    print(f"عکس‌های {a} که مشابهشان در {b} هست: {near_count} از {len(ha)} "
          f"({100 * near_count / max(len(ha), 1):.0f}%)")
    for x, y, d in examples:
        print(f"  نمونه: {x}  ~  {y}  (اختلاف {d}/256)")