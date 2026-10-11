import os, glob, csv, yaml
from collections import Counter

ROOT = r"C:\BalloonProject\dataset\rboflow"
rows = []

def sort_key(n):
    return (0, int(n)) if n.isdigit() else (1, n)

for name in sorted(os.listdir(ROOT), key=sort_key):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        continue

    # اسم کلاس‌ها از data.yaml
    names = "NO data.yaml"
    yp = glob.glob(os.path.join(d, "**", "data.yaml"), recursive=True)
    if yp:
        try:
            with open(yp[0], encoding="utf-8") as f:
                names = yaml.safe_load(f).get("names")
        except Exception as e:
            names = "yaml error"

    splits = Counter()
    areas = []
    n_boxes = 0
    empty = 0
    polygon = False

    for lf in glob.glob(os.path.join(d, "**", "labels", "*.txt"), recursive=True):
        split = os.path.basename(os.path.dirname(os.path.dirname(lf)))
        splits[split] += 1
        with open(lf, encoding="utf-8", errors="ignore") as f:
            lines = [l.split() for l in f if l.strip()]
        if not lines:
            empty += 1
        for p in lines:
            n_boxes += 1
            if len(p) == 5:
                areas.append(float(p[3]) * float(p[4]))
            elif len(p) > 5:
                polygon = True
                xs = [float(v) for v in p[1::2]]
                ys = [float(v) for v in p[2::2]]
                areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))

    areas.sort()
    med = areas[len(areas) // 2] * 100 if areas else 0
    small = (sum(a < 0.01 for a in areas) / len(areas) * 100) if areas else 0

    rows.append([name, str(names), dict(splits), n_boxes, empty,
                 round(med, 2), round(small, 1), polygon])
    print(f"{name:>4} | {str(names)[:60]:60} | {dict(splits)} | boxes={n_boxes} | "
          f"empty={empty} | median_area={med:.2f}% | small<1%={small:.0f}% | polygon={polygon}")

with open(r"C:\BalloonProject\inventory.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["folder", "classes", "splits", "boxes", "empty_labels",
                "median_area_pct", "small_pct", "polygon_labels"])
    w.writerows(rows)
print("\nDone. inventory.csv saved.")