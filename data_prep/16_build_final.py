import csv, shutil
from pathlib import Path

from PIL import Image

MERGED = Path(r"C:\BalloonProject\merged_raw")
SYN = Path(r"C:\BalloonProject\synthetic_v2")
OUT = Path(r"C:\BalloonProject\dataset_final")
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

if OUT.exists():
    shutil.rmtree(OUT)
for kind in ("images", "labels"):
    for split in ("train", "val", "test"):
        (OUT / kind / split).mkdir(parents=True)

stats = {}


def record(group, split, W, H, lines):
    s = stats.setdefault((group, split),
                         {"img": 0, "with_box": 0, "boxes": 0, "small32": 0})
    s["img"] += 1
    if lines:
        s["with_box"] += 1
    for parts in lines:
        long_side = max(float(parts[3]) * W, float(parts[4]) * H)
        s["boxes"] += 1
        if long_side < 32:
            s["small32"] += 1


def read_lines(lp):
    return [l.split() for l in lp.read_text().splitlines() if l.strip()]


# --- عکس‌های واقعی، فقط آن‌هایی که در manifest هستند ---
with (MERGED / "manifest.csv").open(encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))

n_disk = sum(1 for p in (MERGED / "images").iterdir() if p.is_file())
missing = 0
for row in manifest:
    name = row["new_name"]
    src = MERGED / "images" / (name + Path(row["source_file"]).suffix)
    lp = MERGED / "labels" / (name + ".txt")
    if not src.exists() or not lp.exists():
        missing += 1
        continue
    split = SPLIT_MAP[row["split"]]
    shutil.copy2(src, OUT / "images" / split / src.name)
    shutil.copy2(lp, OUT / "labels" / split / lp.name)
    with Image.open(src) as im:
        W, H = im.size
    record("real", split, W, H, read_lines(lp))

# --- عکس‌های مصنوعی، فقط train ---
for p in sorted((SYN / "images").glob("*.jpg")):
    lp = SYN / "labels" / (p.stem + ".txt")
    if not lp.exists():
        continue
    shutil.copy2(p, OUT / "images" / "train" / p.name)
    shutil.copy2(lp, OUT / "labels" / "train" / lp.name)
    with Image.open(p) as im:
        W, H = im.size
    record("synthetic", "train", W, H, read_lines(lp))

(OUT / "data.yaml").write_text(
    f"path: {OUT.as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: red_balloon\n",
    encoding="utf-8",
)

print(f"فایل روی دیسک merged_raw: {n_disk} | ردیف manifest: {len(manifest)} | "
      f"فایل اضافه: {n_disk - len(manifest)} | گم‌شده: {missing}")
print("\nمنبع | بخش | عکس | عکس با بالن | باکس | باکس کوچک‌تر از 32px")
for (group, split), s in sorted(stats.items()):
    print(f"{group:>9} | {split:>5} | {s['img']:>5} | {s['with_box']:>5} | "
          f"{s['boxes']:>5} | {s['small32']:>5}")
print(f"\nدیتاست نهایی: {OUT}")