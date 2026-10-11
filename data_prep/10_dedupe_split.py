import csv, os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\BalloonProject\merged_raw")
NEAR_THRESHOLD = 8  # از 256 بیت؛ سخت‌گیرتر از قبل چون این‌بار می‌خوایم فقط خیلی شبیه‌ها یکی بشن
VAL_FRAC, TEST_FRAC = 0.15, 0.10


def dhash(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


manifest_path = ROOT / "manifest.csv"
with manifest_path.open(encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))

print(f"{len(manifest)} عکس، در حال محاسبه‌ی hash ...")
hashes = []
for i, row in enumerate(manifest):
    img_dir = ROOT / "images"
    matches = list(img_dir.glob(row["new_name"] + ".*"))
    h = dhash(matches[0]) if matches else None
    hashes.append(h)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(manifest)}")

# گروه‌بندی: هر عکس یا عضو یک گروه قبلی میشه یا گروه جدید می‌سازه
group_id = [-1] * len(manifest)
next_group = 0
valid_idx = [i for i, h in enumerate(hashes) if h is not None]
H = np.array([hashes[i] for i in valid_idx])

print("در حال گروه‌بندی ...")
assigned = np.zeros(len(valid_idx), dtype=bool)
for a in range(len(valid_idx)):
    if assigned[a]:
        continue
    dist = (H != H[a]).sum(axis=1)
    same = np.where((dist <= NEAR_THRESHOLD) & ~assigned)[0]
    for b in same:
        group_id[valid_idx[b]] = next_group
        assigned[b] = True
    next_group += 1
    if next_group % 1000 == 0:
        print(f"  {next_group} گروه ساخته شد ...")

for i in range(len(manifest)):
    if group_id[i] == -1:
        group_id[i] = next_group
        next_group += 1
    manifest[i]["group_id"] = group_id[i]

n_images = len(manifest)
n_groups = next_group
print(f"\n{n_images} عکس در {n_groups} گروه ({n_images - n_groups} عکس تکراری/نزدیک حذف از شمارش مجزا)")

# تقسیم train/val/test بر اساس گروه
rng = np.random.default_rng(0)
groups = list(range(n_groups))
rng.shuffle(groups)
n_val = int(n_groups * VAL_FRAC)
n_test = int(n_groups * TEST_FRAC)
val_groups = set(groups[:n_val])
test_groups = set(groups[n_val:n_val + n_test])

for row in manifest:
    g = row["group_id"]
    row["split"] = "valid" if g in val_groups else ("test" if g in test_groups else "train")

with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=manifest[0].keys())
    wr.writeheader()
    wr.writerows(manifest)

counts = {"train": 0, "valid": 0, "test": 0}
for row in manifest:
    counts[row["split"]] += 1
print(f"train={counts['train']} | valid={counts['valid']} | test={counts['test']}")
print("manifest.csv به‌روزرسانی شد (ستون group_id و split اضافه شد)")