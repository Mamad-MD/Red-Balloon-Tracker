from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SETS = {
    "synthetic_v2": (Path(r"C:\BalloonProject\synthetic_v2\images"),
                     Path(r"C:\BalloonProject\synthetic_v2\labels")),
    "merged_raw": (Path(r"C:\BalloonProject\merged_raw\images"),
                   Path(r"C:\BalloonProject\merged_raw\labels")),
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for name, (img_dir, lbl_dir) in SETS.items():
    sizes = Counter()
    boxes_per_img = Counter()
    long_side = []
    for p in img_dir.iterdir():
        if p.suffix.lower() not in EXTS:
            continue
        with Image.open(p) as im:
            W, H = im.size
        sizes[(W, H)] += 1
        lp = lbl_dir / (p.stem + ".txt")
        lines = []
        if lp.exists():
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        boxes_per_img[len(lines)] += 1
        for parts in lines:
            w, h = float(parts[3]) * W, float(parts[4]) * H
            long_side.append(max(w, h))

    print(f"\n=== {name} ===")
    print("رایج‌ترین اندازهٔ عکس‌ها:")
    for (W, H), n in sizes.most_common(8):
        print(f"  {W}x{H}: {n}")
    print("تعداد باکس در هر عکس:", dict(sorted(boxes_per_img.items())))
    if long_side:
        a = np.array(long_side)
        print(f"ضلع بلند باکس (پیکسل): کمینه={a.min():.0f} | "
              f"میانه={np.median(a):.0f} | بیشینه={a.max():.0f}")