import csv
from pathlib import Path

import cv2
import numpy as np

MERGED = Path(r"C:\BalloonProject\merged_raw")
OUT_SKY_CSV = Path(r"C:\BalloonProject\sky_candidates.csv")
OUT_REVIEW = Path(r"C:\BalloonProject\review\sky_numbered.jpg")


def sky_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)

    # آسمون آبی واقعی: hue محدود به بازه‌ی آبی، اشباع متوسط به بالا (نه رنگ‌پریده مثل اسفنج)
    blue_sky = (h >= 95) & (h <= 120) & (s >= 40) & (s <= 200) & (v >= 90)
    # آسمون ابری/سفید: اشباع خیلی کم و روشنایی بالا
    overcast = (s < 25) & (v > 140)
    sky_mask = blue_sky | overcast
    sky_ratio = float(sky_mask.mean())

    # یکنواختی رنگ: انحراف‌معیار hue روی پیکسل‌های آسمانی باید کم باشه
    if sky_mask.sum() > 100:
        hue_std = float(h[sky_mask].std())
    else:
        hue_std = 999

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_std = float(gray.std())  # آسمون تقریباً صاف؛ بافت (خط، الگو) این عدد رو بالا می‌بره
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255)

    is_sky = (sky_ratio > 0.85 and hue_std < 12 and
              texture_std < 25 and edge_density < 0.02)
    return is_sky, round(sky_ratio, 2), round(hue_std, 1), round(texture_std, 1), round(edge_density, 3)


with (MERGED / "manifest.csv").open(encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))

sky_rows = []
for i, row in enumerate(manifest):
    if row["n_boxes"] != "0":
        continue
    img_dir = MERGED / "images"
    matches = list(img_dir.glob(row["new_name"] + ".*"))
    if not matches:
        continue
    img = cv2.imdecode(np.fromfile(str(matches[0]), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    is_sky, sky_ratio, hue_std, tex_std, edge_d = sky_score(img)
    if is_sky:
        sky_rows.append({"file": matches[0].name, "sky_ratio": sky_ratio,
                          "hue_std": hue_std, "texture_std": tex_std, "edge_density": edge_d})
    if (i + 1) % 3000 == 0:
        print(f"  {i+1}/{len(manifest)}")

with OUT_SKY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=["file", "sky_ratio", "hue_std", "texture_std", "edge_density"])
    wr.writeheader()
    wr.writerows(sky_rows)

print(f"\nعکس‌های آسمانی پیدا شده: {len(sky_rows)}")

# کانتکت‌شیت شماره‌دار برای بررسی چشمی
T = 140
tiles = []
img_dir = MERGED / "images"
for idx, row in enumerate(sky_rows):
    img = cv2.imdecode(np.fromfile(str(img_dir / row["file"]), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    tile = cv2.resize(img, (T, T))
    cv2.putText(tile, str(idx), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    tiles.append(tile)

per_row = 12
pad = per_row - (len(tiles) % per_row or per_row)
tiles += [np.zeros((T, T, 3), np.uint8)] * pad
rows_img = [np.hstack(tiles[j:j+per_row]) for j in range(0, len(tiles), per_row)]
cv2.imwrite(str(OUT_REVIEW), np.vstack(rows_img))
print(f"کانتکت‌شیت ذخیره شد: {OUT_REVIEW}")